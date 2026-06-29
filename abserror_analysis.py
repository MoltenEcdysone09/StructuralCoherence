import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
# from matplotlib.patches import Patch
# import matplotlib.lines as mlines
# from scipy import stats
# from scipy.stats import spearmanr
# from sklearn.linear_model import HuberRegressor
# from sklearn.exceptions import ConvergenceWarning
# import statsmodels.formula.api as smf
# import warnings

# =====================================================================
# 1. STYLE & DESIGN PATTERNS (Nord Theme)
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
            "grid.color": NORD_COLORS["gray"],
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def preprocess_data(filepath):
    """
    Loads and normalizes the coherence data.
    """
    print("Loading and preprocessing data...")
    df = pd.read_parquet(filepath)
    # Dropping some some un-necessary columns
    df = df.drop(
        columns=[
            "AbsMeanCohVal",
            "AbsMedianCohVal",
            "CohMatMean",
            "CohMatMedian",
            "AbsMedianWalkVal",
            "MedianCoh",
            "PreSplitMedianCoh",
        ]
    )
    # Renaming the "Groups" to "Teams", MeanCoh to StrcutCoh and WalkVal to MeanComm
    df = df.rename(
        columns={
            "NumGroups": "NumTeams",
            "NumPreSplitGroups": "NumPreSplitTeams",
            "MeanCoh": "StructCoh",
            "PreSplitMeanCoh": "PreSplitStructCoh",
            "AbsMeanWalkVal": "NormMeanComm",
        }
    )

    df["BaseNet"] = df["MAN_code"].astype(str) + "_" + df["EdgeString"].astype(str)
    df["ScaleStr"] = df["Scale"].astype(str) + "x"
    # print(df)
    # print(df.columns)
    # print(df.dtypes)

    # Getting the refence values of the maximum density versions
    reference_df = df[df["Density"] == 1.0]
    # print(reference_df)

    # Grouping ensures a stable mean baseline if there are multiple replicates at max density
    group_cols = ["BaseNet", "Scale", "NetType", "SelfActivation"]

    # Creating lookup dicts for the metrics
    ref_map_teams = reference_df.groupby(group_cols)["NumTeams"].mean().to_dict()
    ref_map_psteams = (
        reference_df.groupby(group_cols)["NumPreSplitTeams"].mean().to_dict()
    )
    ref_map_structcoh = reference_df.groupby(group_cols)["StructCoh"].mean().to_dict()
    ref_map_psstructcoh = (
        reference_df.groupby(group_cols)["PreSplitStructCoh"].mean().to_dict()
    )

    def calculate_differences(row):
        key = (row["BaseNet"], row["Scale"], row["NetType"], row["SelfActivation"])

        baseline_teams = ref_map_teams.get(key, np.nan)
        baseline_pre_teams = ref_map_psteams.get(key, np.nan)
        baseline_structcoh = ref_map_structcoh.get(key, np.nan)
        baseline_pre_structcoh = ref_map_psstructcoh.get(key, np.nan)

        return pd.Series(
            {
                "AE_NumTeams": np.abs(row["NumTeams"] - baseline_teams)
                if pd.notna(baseline_teams)
                else np.nan,
                "AE_NumPreSplitTeams": np.abs(
                    row["NumPreSplitTeams"] - baseline_pre_teams
                )
                if pd.notna(baseline_pre_teams)
                else np.nan,
                "Norm_AE_NumTeams": np.abs(row["NumTeams"] - baseline_teams)
                / row["NumNodes"]
                if pd.notna(baseline_teams)
                else np.nan,
                "Norm_AE_NumPreSplitTeams": np.abs(
                    row["NumPreSplitTeams"] - baseline_pre_teams
                )
                / row["NumNodes"]
                if pd.notna(baseline_pre_teams)
                else np.nan,
                "AE_StructCoh": np.abs(row["StructCoh"] - baseline_structcoh)
                if pd.notna(baseline_structcoh)
                else np.nan,
                "AE_PreSplitStructCoh": np.abs(
                    row["PreSplitStructCoh"] - baseline_pre_structcoh
                )
                if pd.notna(baseline_pre_structcoh)
                else np.nan,
                "Norm_AE_StructCoh": np.abs(row["StructCoh"] - baseline_structcoh) / 2
                if pd.notna(baseline_structcoh)
                else np.nan,
                "Norm_AE_PreSplitStructCoh": np.abs(
                    row["PreSplitStructCoh"] - baseline_pre_structcoh
                )
                / 2
                if pd.notna(baseline_pre_structcoh)
                else np.nan,
            }
        )

    diff_cols = df.apply(calculate_differences, axis=1)
    df = pd.concat([df, diff_cols], axis=1)
    # print(df)
    # print(df.columns)
    # print(df.dtypes)
    # print(df.shape)

    # df = df.dropna(
    #     subset=[
    #         "AE_NumTeams",
    #         "AE_NumPreSplitTeams",
    #         "Norm_AE_NumTeams",
    #         "Norm_AE_NumPreSplitTeams",
    #         "AE_StructCoh",
    #         "AE_PreSplitStructCoh",
    #         "Norm_AE_StructCoh",
    #         "Norm_AE_PreSplitStructCoh",
    #         "NormMeanComm",
    #     ]
    # )
    # print(df.shape)

    return df


def calculate_cae(agg_df, metric):
    """Returns the pre-calculated sum of the metric (CAE)."""
    return agg_df[(metric, "sum")]


def calculate_ratio(agg_df, metric):
    """Vectorized calculation of the curve-to-line Ratio."""
    x_0 = agg_df[("Density", "first")]
    x_n = agg_df[("Density", "last")]
    x_sum = agg_df[("Density", "sum")]
    N = agg_df[("Density", "count")]

    y_0 = agg_df[(metric, "first")]
    y_n = agg_df[(metric, "last")]
    cae = agg_df[(metric, "sum")]

    # Calculate difference in x; handle cases where x_n == x_0 to avoid division by zero
    dx = x_n - x_0

    # Calculate slope (m) where dx is non-zero, otherwise 0
    m = np.where(dx != 0, (y_n - y_0) / dx, 0.0)

    # Algebraic reduction of the sum of a linear progression evaluated at specific x coordinates
    baseline_sum = (N * y_0) + (m * (x_sum - (N * x_0)))

    # Calculate Ratio: mask out division-by-zero or groups with fewer than 2 points
    valid_mask = (dx != 0) & (baseline_sum != 0) & (N >= 2)
    ratio = np.where(valid_mask, cae / baseline_sum, np.nan)

    return ratio


def calculate_cae_metrics(df):
    """
    Calculates the Cumulative Absolute Error (CAE) and the curve-to-line Ratio
    based on discrete sums for each replicate.

    Parameters:
    - df: The preprocessed DataFrame.
    - target_metrics: List of column names to calculate CAE and Ratio for.

    Returns:
    - A new DataFrame containing replicate-wise CAE and Ratio for each metric.
    """
    TARGET_METRICS = [
        "AE_NumTeams",
        "AE_NumPreSplitTeams",
        "Norm_AE_NumTeams",
        "Norm_AE_NumPreSplitTeams",
        "AE_StructCoh",
        # "AE_PreSplitStructCoh",
        "Norm_AE_StructCoh",
        # "Norm_AE_PreSplitStructCoh",
    ]

    # Use "Rep" as that is the actual column name in your DataFrame schema
    group_cols = ["BaseNet", "Scale", "NetType", "SelfActivation", "Rep"]

    # Sort by grouping columns and strictly by Density to ensure chronological X-axis
    df_sorted = df.sort_values(by=group_cols + ["Density"]).copy()

    # 2. Build the aggregation dictionary for a single pass
    agg_dict = {"Density": ["first", "last", "sum", "count"]}
    for m in TARGET_METRICS:
        agg_dict[m] = ["first", "last", "sum"]

    # 3. Execute the C-optimized groupby aggregation
    agg_df = df_sorted.groupby(group_cols).agg(agg_dict)

    # 4. Initialize the final output DataFrame using the multi-index from the groupby
    results_df = pd.DataFrame(index=agg_df.index)

    # 5. Apply the vectorized functions for each metric
    for m in TARGET_METRICS:
        results_df[f"{m.replace('AE_', 'CAE_')}"] = calculate_cae(agg_df, m)
        results_df[f"Ratio_{m}"] = calculate_ratio(agg_df, m)

    # FUnction to calucalte spearman correlation
    def compute_spearman_for_group(group):
        metrics_dict = {}

        # Define the explicit pairs you want to correlate
        correlation_pairs = [
            ("AE_NumTeams", "AE_StructCoh"),
            ("Norm_AE_NumTeams", "Norm_AE_StructCoh"),
        ]

        for m1, m2 in correlation_pairs:
            # Drop NaNs safely for this specific pair of metrics
            valid_data = group[[m1, m2]].dropna()

            # Spearman requires variance in BOTH arrays.
            # We check that both metrics have at least 2 unique values within the trajectory.
            if (
                len(valid_data) >= 2
                and valid_data[m1].nunique() > 1
                and valid_data[m2].nunique() > 1
            ):
                rho, pval = spearmanr(valid_data[m1], valid_data[m2])

                # Name the output columns to explicitly show which metrics were compared
                metrics_dict[f"Spearman_Rho_{m1}_vs_{m2}"] = rho
                metrics_dict[f"Spearman_Pval_{m1}_vs_{m2}"] = pval
            else:
                # Fallback for perfectly flat trajectories
                metrics_dict[f"Spearman_Rho_{m1}_vs_{m2}"] = np.nan
                metrics_dict[f"Spearman_Pval_{m1}_vs_{m2}"] = np.nan

        return pd.Series(metrics_dict)

    spearman_df = df_sorted.groupby(group_cols).apply(compute_spearman_for_group)
    results_df = pd.concat([results_df, spearman_df], axis=1)

    return results_df.reset_index()


#########################################################################################
#### PLOTTING
#########################################################################################


def plot_metric_trajectories_panel_F(df, target_metric, save_dir, nord_colors):
    """
    Generates side-by-side Seaborn line plots for all MAN_codes within a specific
    Scale, NetType, and SelfActivation combination.
    Shares the Y-axis for direct comparison and uses a single, unified colorbar.
    """
    set_global_nord_style()
    # 1. Build the sequential, unidirectional colormap
    # Mapping: 0.0 -> Light Nord Gray (#d8dee9), 1.0 -> Nord Blue
    # cmap_colors = ["#3b4252", nord_colors["cyan"]]
    cmap_colors = [nord_colors["green"], nord_colors["orange"]]
    structcoh_cmap = LinearSegmentedColormap.from_list(
        "AbsStructCoh_Global", cmap_colors
    )

    # Defining metric labels
    metric_labels = {
        "AE_NumTeams": r"$| \Delta \mathrm{Teams} |$",
        "Norm_AE_NumTeams": r"Norm. $| \Delta \mathrm{Teams} |$",
        "AE_StructCoh": r"$| \Delta C_{\mathrm{struct}} |$",
        "Norm_AE_StructCoh": r"Norm. $| \Delta C_{\mathrm{struct}} |$",
        "AE_NumPreSplitTeams": r"AE(Pre-Split Teams)",
        "Norm_AE_NumPreSplitTeams": r"Norm. AE(Pre-Split Teams)",
        "AE_PreSplitStructCoh": r"AE(Pre-Split $C_{struct}$)",
        "Norm_AE_PreSplitStructCoh": r"Norm. AE(Pre-Split $C_{struct}$)",
    }

    # Normalize the mappable object exactly to the [0.0, 1.0] domain
    norm = Normalize(vmin=0.0, vmax=1.0)
    mappable = cm.ScalarMappable(norm=norm, cmap=structcoh_cmap)

    # 2. Extract the reference StructCoh (at Density == 1.0) for every BaseNet
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["MAN_code", "Scale", "NetType", "SelfActivation", "BaseNet"])[
            "StructCoh"
        ]
        .mean()
        .reset_index()
    )

    # Calculate the absolute value for the sequential mapping
    ref_df["AbsStructCoh"] = ref_df["StructCoh"].abs()

    # 3. Group by the primary plot parameters (excluding MAN_code to group them together)
    plot_groups = ["Scale", "NetType", "SelfActivation"]

    for group_keys, group_data in df.groupby(plot_groups):
        scale, net_type, sa = group_keys

        # Identify all unique MAN_codes in this specific combination
        man_codes = sorted(group_data["MAN_code"].unique())
        n_plots = len(man_codes)

        if n_plots == 0:
            continue

        # 4. Dynamically route output directories based on the NetType
        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        # 5. Initialize the side-by-side figure panel
        # Width scales dynamically based on the number of MAN_codes.
        # sharey=True ensures the Y-axis scale is locked across all subplots.
        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(4.8 * n_plots + 0.8, 3.25),
            sharey=True,
            sharex=True,
            gridspec_kw={"wspace": 0.20},
        )

        # Ensure axes is iterable even if there is only one MAN_code
        if n_plots == 1:
            axes = [axes]

        # 6. Iterate through each MAN_code and its corresponding subplot axis
        for ax, man_code in zip(axes, man_codes):
            mc_data = group_data[group_data["MAN_code"] == man_code]

            # Isolate the reference Structural Coherence values for the current subplot
            group_refs = ref_df[
                (ref_df["MAN_code"] == man_code)
                & (ref_df["Scale"] == scale)
                & (ref_df["NetType"] == net_type)
                & (ref_df["SelfActivation"] == sa)
            ]

            # Map each BaseNet to its specific absolute color from the global map
            basenet_colors = {}
            for _, row in group_refs.iterrows():
                basenet_colors[row["BaseNet"]] = mappable.to_rgba(row["AbsStructCoh"])

            # Plot the trajectory
            sns.lineplot(
                data=mc_data,
                x="Density",
                y=target_metric,
                hue="BaseNet",
                palette=basenet_colors,
                linewidth=3.0,
                marker="o",
                markersize=5,
                ax=ax,
                errorbar="sd",
                err_style="bars",
                err_kws={
                    "capsize": 4,
                    "elinewidth": 2.0,
                    "capthick": 2.0,
                },
            )

            # Format individual subplot
            ax.set_xlabel("Density")
            ax.set_title(f"{man_code}")

            ax.tick_params(axis="y", labelleft=True)

            # Only set the Y-label on the first (left-most) plot to reduce clutter
            if ax == axes[0]:
                # Fallback to the replace method if the metric isn't in the dictionary
                clean_ylabel = metric_labels.get(
                    target_metric, target_metric.replace("_", " ")
                )
                ax.set_ylabel(clean_ylabel)
            else:
                ax.set_ylabel("")

            # Remove Seaborn's default legend
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # 7. Add unified Figure formatting
        fig.suptitle(f"Scale: {scale}x | {sa} | {net_type}", y=1.05, fontsize=16)

        # Add the colorbar attached to the list of axes to place it globally on the right
        # cbar = fig.colorbar(mappable, ax=axes)
        # cbar.set_label(r"|$C_{struct}$| at Density = 1.0")

        # ---------------------------------------------------------
        # COLORBAR & ANNOTATED PIPED LEGEND
        # ---------------------------------------------------------
        # Add the colorbar attached to the list of axes
        cbar = fig.colorbar(mappable, ax=axes, pad=0.05)

        # Flip the ticks and the main label to the LEFT side of the colorbar
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        # cbar.set_label(r"$|C_{struct}|_{basal}$")
        cbar.set_label(r"${|C_{struct}|}_{basal}$")

        # Isolate all unique BaseNets plotted in this specific panel
        panel_refs = (
            ref_df[
                (ref_df["Scale"] == scale)
                & (ref_df["NetType"] == net_type)
                & (ref_df["SelfActivation"] == sa)
                & (ref_df["MAN_code"].isin(man_codes))
            ]
            .drop_duplicates(subset=["BaseNet"])
            .sort_values(by="AbsStructCoh")
        )

        # Calculate evenly spaced Y-coordinates to prevent text overlap
        n_labels = len(panel_refs)
        if n_labels > 0:
            # Create an evenly spaced vertical grid from 0.02 to 0.98 for the text labels
            y_text_positions = np.linspace(0.02, 0.98, n_labels)

            # Draw the clean routing pipes and text labels
            for i, (_, row) in enumerate(panel_refs.iterrows()):
                y_val = row["AbsStructCoh"]
                basenet = row["BaseNet"]
                y_text = y_text_positions[i]

                # Fetch the exact RGBA color mapping for this network value
                line_color = mappable.to_rgba(y_val)

                # 1.0  -> 1.08: Short, crisp horizontal exit stub from colorbar
                # 1.08 -> 2.68: Massive slanted transition routing spanning 1.6 units
                # 2.68 -> 3.68: Long horizontal entry stub leading into text
                x_pipe = [1.0, 1.08, 2.68, 3.68]
                y_pipe = [y_val, y_val, y_text, y_text]

                # Plot the conduit line using its native mapped color
                cbar.ax.plot(
                    x_pipe,
                    y_pipe,
                    color=line_color,
                    # alpha=0.85,
                    linewidth=1.75,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    clip_on=False,
                )

                # Anchor dot matching the path color on the colorbar edge
                cbar.ax.plot(
                    1.0,
                    y_val,
                    marker=".",
                    markersize=5,
                    color=line_color,
                    clip_on=False,
                )

                # Clean the text label formatting: replace "_" with "-"
                clean_label = basenet.replace("_", "-")
                final_text = f"{clean_label} ({row['AbsStructCoh']:.2f})"

                # Offset the text labels to align with the extended final horizontal segment
                cbar.ax.text(
                    3.80,
                    y_text,
                    final_text,
                    fontsize=12,
                    color=nord_colors["dark"],
                    va="center",
                    ha="left",
                    clip_on=False,
                )

        # 8. Execute the Naming Scheme & Save
        # Filename reflects that it contains a panel of multiple MAN_codes
        filename = f"Panel_{scale}_{sa}_{net_type}_{target_metric}VsDensity"

        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )

        plt.close(fig)


def plot_ratio_heatmaps_F(cae_df, df, target_metric, save_dir, nord_colors):
    """
    Generates vertically segmented heatmaps for CAE metrics across Scales and BaseNets.
    Segments heatmaps by MAN_code with clear gaps, sharing a common X-axis.

    Colormap: Sequential palette mapping 0 to the lightest tone and the maximum data value
              to the darkest tone (Nord-inspired dark charcoal slate).
    Normalization: Standard linear scaling (TwoSlopeNorm centered at 1.0 is completely removed).
    Significance Annotations:
        - Significant Decrease (p < 0.05, left > right): Light right arrow (->)
        - Significant Increase (p < 0.05, left < right): Light left arrow (<-)
        - Non-Significant Transition: Thin, standard line with no arrowheads.
    """
    # Define clean LaTeX-formatted titles/labels sourced from tracking functions
    metric_labels = {
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_StructCoh": r"Cumulative Norm. $| \Delta C_{\mathrm{struct}} |$",
        "CAE_NumTeams": r"Cumulative $| \Delta \mathrm{Teams} |$",
        "CAE_StructCoh": r"Cumulative $| \Delta C_{\mathrm{struct}} |$",
    }

    # 1. Extract reference coherence at Density = 1.0 for ordering rows
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["NetType", "SelfActivation", "MAN_code", "BaseNet"])["StructCoh"]
        .mean()
        .reset_index()
    )
    ref_df["AbsStructCoh"] = ref_df["StructCoh"].abs()

    # 2. Build a Sequential Colormap: 0.0 is the lightest, max is the darkest
    # Maps from a clean light background grey up to your signature Nord dark gray/charcoal
    cmap_colors = [nord_colors["blue"], nord_colors["green"]]
    sequential_cmap = mcolors.LinearSegmentedColormap.from_list(
        "Sequential_CAE_Heatmap", cmap_colors
    )
    sequential_cmap.set_bad(color="#1a1c23")  # Clear deep tint for missing blocks

    # 3. Iterate through NetType and SelfActivation combinations
    plot_groups = ["NetType", "SelfActivation"]
    for group_keys, group_cae_data in cae_df.groupby(plot_groups):
        net_type, sa = group_keys

        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        group_cae_copy = group_cae_data.copy()
        if "MAN_code" not in group_cae_copy.columns:
            group_cae_copy["MAN_code"] = (
                group_cae_copy["BaseNet"].astype(str).str.split("_").str[0]
            )

        # Aggregate replicate data by taking the mean metric across trials
        agg_data = (
            group_cae_copy.groupby(["MAN_code", "BaseNet", "Scale"])[target_metric]
            .mean()
            .reset_index()
        )

        sorted_scales = sorted(agg_data["Scale"].unique())
        scale_labels = [f"{s}x" for s in sorted_scales]
        man_codes = sorted(agg_data["MAN_code"].unique())
        n_mancodes = len(man_codes)

        if n_mancodes == 0:
            continue

        facet_refs = ref_df[
            (ref_df["NetType"] == net_type) & (ref_df["SelfActivation"] == sa)
        ]

        mancode_networks = {}
        height_ratios = []
        for mc in man_codes:
            mc_refs = facet_refs[facet_refs["MAN_code"] == mc].sort_values(
                by="AbsStructCoh"
            )
            networks = [
                n
                for n in mc_refs["BaseNet"].tolist()
                if n in agg_data["BaseNet"].values
            ]
            mancode_networks[mc] = networks
            height_ratios.append(max(1, len(networks)))

        # 4. Standard Linear Normalization (Centering at 1.0 removed entirely)
        vmin_val = 0.0  # Forces the scale floor to hit your absolute baseline zero
        vmax_val = agg_data[target_metric].max()
        norm = mcolors.Normalize(vmin=vmin_val, vmax=vmax_val)

        # Create vertical stack of subplots
        fig, axes = plt.subplots(
            nrows=n_mancodes,
            ncols=1,
            figsize=(4.0, 0.45 * sum(height_ratios) + 1.5),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.07},
        )

        if n_mancodes == 1:
            axes = [axes]

        # Plot each heatmap segment
        for idx, (ax, mc) in enumerate(zip(axes, man_codes)):
            mc_data = agg_data[agg_data["MAN_code"] == mc]

            matrix = mc_data.pivot(
                index="BaseNet", columns="Scale", values=target_metric
            )
            ordered_rows = mancode_networks[mc]
            matrix = matrix.reindex(index=ordered_rows, columns=sorted_scales)
            matrix.index = [idx_str.replace("_", "-") for idx_str in matrix.index]

            sns.heatmap(
                data=matrix,
                ax=ax,
                cmap=sequential_cmap,
                norm=norm,
                cbar=False,
                annot=True,
                fmt=".3f",
                linewidths=0.8,
                linecolor="white",
                xticklabels=scale_labels,
                yticklabels=True,
                mask=matrix.isna(),
            )

            # -----------------------------------------------------------------
            # NON-PARAMETRIC TRANSITION ANNOTATIONS (REDUCED WEIGHT & FLOW LINES)
            # -----------------------------------------------------------------
            raw_mc_data = group_cae_copy[group_cae_copy["MAN_code"] == mc]

            for s_idx in range(len(sorted_scales) - 1):
                scale_left = sorted_scales[s_idx]
                scale_right = sorted_scales[s_idx + 1]

                for r_idx, basenet_dash in enumerate(matrix.index):
                    basenet_raw = basenet_dash.replace("-", "_")

                    rep_pool_left = (
                        raw_mc_data[
                            (raw_mc_data["BaseNet"] == basenet_raw)
                            & (raw_mc_data["Scale"] == scale_left)
                        ][target_metric]
                        .dropna()
                        .values
                    )

                    rep_pool_right = (
                        raw_mc_data[
                            (raw_mc_data["BaseNet"] == basenet_raw)
                            & (raw_mc_data["Scale"] == scale_right)
                        ][target_metric]
                        .dropna()
                        .values
                    )

                    if len(rep_pool_left) > 0 and len(rep_pool_right) > 0:
                        _, p_val_dec = mannwhitneyu(
                            rep_pool_left, rep_pool_right, alternative="greater"
                        )
                        _, p_val_inc = mannwhitneyu(
                            rep_pool_left, rep_pool_right, alternative="less"
                        )

                        x_coord = s_idx + 1
                        y_coord = r_idx + 0.5

                        if p_val_dec < 0.05:
                            # Significant decrease: Subtle right-pointing arrow
                            ax.annotate(
                                "",
                                xy=(x_coord + 0.12, y_coord),
                                xytext=(x_coord - 0.12, y_coord),
                                arrowprops=dict(
                                    arrowstyle="-|>",
                                    color=nord_colors["dark"],
                                    lw=2.5,  # Softened linewidth
                                    mutation_scale=30,  # Scaled down head size
                                    zorder=5,
                                ),
                            )
                        elif p_val_inc < 0.05:
                            # Significant increase: Subtle left-pointing arrow
                            ax.annotate(
                                "",
                                xy=(x_coord - 0.12, y_coord),
                                xytext=(x_coord + 0.12, y_coord),
                                arrowprops=dict(
                                    arrowstyle="-|>",
                                    color=nord_colors["dark"],
                                    lw=2.5,  # Softened linewidth
                                    mutation_scale=30,  # Scaled down head size
                                    zorder=5,
                                ),
                            )
                        else:
                            # Non-significant shift: Clean horizontal line bridge without arrowheads
                            ax.plot(
                                [x_coord - 0.08, x_coord + 0.08],
                                [y_coord, y_coord],
                                color=nord_colors["dark"],
                                linewidth=1.5,  # Thin and non-intrusive
                                linestyle="-",
                                solid_capstyle="round",
                                zorder=4,
                            )

            ax.set_ylabel(mc, rotation=0, labelpad=45, va="center")
            ax.set_xlabel("")
            ax.tick_params(axis="y", left=False)

            if idx < n_mancodes - 1:
                ax.tick_params(axis="x", bottom=False)

        # Global layout title formatting
        clean_name = metric_labels.get(target_metric, target_metric.replace("_", " "))
        fig.suptitle(f"{clean_name} Matrix | {sa} | {net_type}", y=0.96)

        plt.tight_layout(rect=[0, 0, 0.75, 0.93])

        # Position colorbar axis channel
        cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.68])

        # Render 3-point sequential colorbar matching limits dynamically
        sm = plt.cm.ScalarMappable(cmap=sequential_cmap, norm=norm)
        v_min = norm.vmin
        v_max = norm.vmax
        v_mid = (v_min + v_max) / 2.0
        three_point_ticks = [v_min, v_mid, v_max]

        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=three_point_ticks)
        cbar.ax.set_yticklabels([f"{v_min:.2f}", f"{v_mid:.2f}", f"{v_max:.2f}"])

        # Colorbar title dynamically tracks the LaTeX name dictionary
        cbar.set_label(clean_name)

        # Save systems
        filename = f"Heatmap_{sa}_{net_type}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )

        plt.close(fig)


def plot_scale_trajectories_with_stats_F(
    cae_df, df, target_metric, save_dir, nord_colors
):
    """
    Plots target CAE metrics across scales using BaseNet architectures as structural anchors.

    X-axis: ScaleStr (10x, 30x, 50x)
    Y-axis: Target Metric
    Lines: Tracks individual network trajectories across scales with standard deviation error bars.
    Background: Synchronized, color-matched jittered replicate scattering to display raw variance.
    """
    set_global_nord_style()

    metric_labels = {
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_StructCoh": r"Cumulative Norm. $| \Delta C_{\mathrm{struct}} |$",
        "CAE_NumTeams": r"Cumulative $| \Delta \mathrm{Teams} |$",
        "CAE_StructCoh": r"Cumulative $| \Delta C_{\mathrm{struct}} |$",
    }

    # 1. Build the colormap for the structural coherence tracks
    cmap_colors = [nord_colors["green"], nord_colors["orange"]]
    structcoh_cmap = mcolors.LinearSegmentedColormap.from_list(
        "AbsStructCoh_Scale", cmap_colors
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    mappable = cm.ScalarMappable(norm=norm, cmap=structcoh_cmap)

    # 2. Extract reference absolute coherence at Density = 1.0
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["NetType", "SelfActivation", "BaseNet"])["StructCoh"]
        .mean()
        .reset_index()
    )
    ref_df["AbsStructCoh"] = ref_df["StructCoh"].abs()

    # Merge to link replicates to their network's structural coherence property
    plot_df = pd.merge(cae_df, ref_df, on=["NetType", "SelfActivation", "BaseNet"])

    if "ScaleStr" not in plot_df.columns:
        plot_df["ScaleStr"] = plot_df["Scale"].astype(str) + "x"
    unique_scales = sorted(plot_df["Scale"].unique())
    scale_labels = [f"{s}x" for s in unique_scales]

    # Calculate network means and standard deviations across replicates
    network_stats = (
        plot_df.groupby(
            ["NetType", "SelfActivation", "BaseNet", "ScaleStr", "AbsStructCoh"]
        )[target_metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Create an integer mapping for the discrete X-axis to position jittered points cleanly
    scale_x_map = {label: i for i, label in enumerate(scale_labels)}

    for (net_type, sa), group_data in plot_df.groupby(["NetType", "SelfActivation"]):
        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        facet_stats = network_stats[
            (network_stats["NetType"] == net_type)
            & (network_stats["SelfActivation"] == sa)
        ]
        facet_raw = group_data[
            (group_data["NetType"] == net_type) & (group_data["SelfActivation"] == sa)
        ]

        if facet_stats.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))

        # 3. Synchronized Loop: Color matching lines, error bars, and replicate points per network
        for basenet, net_stats_group in facet_stats.groupby("BaseNet"):
            net_stats_sorted = net_stats_group.sort_values(
                by="ScaleStr", key=lambda x: x.str.replace("x", "").astype(int)
            )

            # Extract color mapping metrics
            coherence_val = net_stats_sorted["AbsStructCoh"].iloc[0]
            thread_color = mappable.to_rgba(coherence_val)

            # Isolate raw replicate rows for this network to add matching background scatter
            net_raw = facet_raw[facet_raw["BaseNet"] == basenet]

            # --- A. Plot Color-Matched Jittered Background Replicates ---
            # Map categorical scale names into numerical X-coordinates
            x_indices = net_raw["ScaleStr"].map(scale_x_map).values
            # Generate deterministic jitter bounding coordinates
            rng = np.random.default_rng(seed=hash(basenet) % (2**32))
            jitter = rng.uniform(-0.12, 0.12, size=len(x_indices))
            x_jittered = x_indices + jitter

            ax.scatter(
                x_jittered,
                net_raw[target_metric],
                color=thread_color,
                alpha=0.52,  # Keep translucent to stay in background
                s=15,
                edgecolor="none",
                zorder=1,
            )

            # --- B. Plot Individual Trajectory Lines with Standard Deviation Bars ---
            ax.errorbar(
                x=net_stats_sorted["ScaleStr"],
                y=net_stats_sorted["mean"],
                yerr=net_stats_sorted["std"].fillna(0),  # Guard against lone replicates
                color=thread_color,
                alpha=1.0,  # Full opacity for contrast
                linewidth=2.0,
                marker="o",
                markersize=5,
                capsize=4,  # Clear cap width for error visualization
                elinewidth=1.5,
                capthick=1.5,
                zorder=2,
            )

        # 4. Labels & Title Setup
        ax.set_xticks(range(len(scale_labels)))
        ax.set_xticklabels(scale_labels)
        ax.set_xlabel("Scale Variant")
        ax.set_ylabel(metric_labels[target_metric])
        ax.set_title(
            f"{target_metric.replace('_', ' ')} across Scales | {sa} | {net_type}",
            pad=20,
            fontsize=14,
        )

        # Dynamic y-limit configuration with safety padding
        y_max = facet_raw[target_metric].max()
        ax.set_ylim(bottom=-y_max * 0.03, top=y_max * 1.05)

        plt.tight_layout(rect=[0, 0, 0.70, 0.95])

        # 5. Render Piped Routing Legend Panel
        cbar_ax = fig.add_axes([0.80, 0.15, 0.025, 0.65])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.set_label(r"${|C_{struct}|}_{basal}$")

        panel_refs = (
            ref_df[
                (ref_df["NetType"] == net_type)
                & (ref_df["SelfActivation"] == sa)
                & (ref_df["BaseNet"].isin(facet_stats["BaseNet"].unique()))
            ]
            .drop_duplicates(subset=["BaseNet"])
            .sort_values(by="AbsStructCoh")
        )

        n_labels = len(panel_refs)
        if n_labels > 0:
            y_text_positions = np.linspace(0.02, 0.98, n_labels)

            for i, (_, row) in enumerate(panel_refs.iterrows()):
                y_val = row["AbsStructCoh"]
                basenet = row["BaseNet"]
                y_text = y_text_positions[i]
                line_color = mappable.to_rgba(y_val)

                x_pipe = [1.0, 1.10, 2.80, 4.00]
                y_pipe = [y_val, y_val, y_text, y_text]

                cbar.ax.plot(
                    x_pipe,
                    y_pipe,
                    color=line_color,
                    linewidth=1.5,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    clip_on=False,
                )
                cbar.ax.plot(
                    1.0,
                    y_val,
                    marker=".",
                    markersize=4,
                    color=line_color,
                    clip_on=False,
                )

                clean_label = basenet.replace("_", "-")
                cbar.ax.text(
                    4.20,
                    y_text,
                    f"{clean_label} ({y_val:.2f})",
                    color=nord_colors["dark"],
                    va="center",
                    ha="left",
                    clip_on=False,
                )

        # 6. Save Systems
        filename = f"Scale_Trajectory_{sa}_{net_type}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_sa_vs_ns_paired_10x_F(cae_df, df, target_metric, save_dir, nord_colors):
    """
    Plots 10x scale paired boxes comparing SA and NS side-by-side.

    Sorting: Networks are grouped by MAN_code, and within each family block,
             sorted sequentially by the difference in Absolute Structural Coherence
             at Density = 1.0 between the SA and NS versions (|C_struct,SA| - |C_struct,NS|).
    Significance: Default statannotations star system (*, **, ***, ns) inside the grid headroom.
    Legend: Bounded external axis frame on the top right.
    """
    set_global_nord_style()

    df_10x = cae_df[cae_df["Scale"] == 10].copy()
    if df_10x.empty:
        return

    if "MAN_code" not in df_10x.columns:
        df_10x["MAN_code"] = df_10x["BaseNet"].astype(str).str.split("_").str[0]

    metric_labels = {
        "CAE_NumTeams": r"Cumulative $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
    }
    clean_ylabel = metric_labels.get(target_metric, target_metric.replace("_", " "))

    # 1. Extract reference background structural coherence configurations at Density == 1.0
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["NetType", "SelfActivation", "MAN_code", "BaseNet"])["StructCoh"]
        .mean()
        .reset_index()
    )

    # Pivot ref_df to compute the precise difference in absolute structural coherence between conditions
    ref_pivot = ref_df.pivot(
        index=["NetType", "MAN_code", "BaseNet"],
        columns="SelfActivation",
        values="StructCoh",
    ).reset_index()

    ref_pivot["SA"] = ref_pivot["SA"].fillna(0)
    ref_pivot["NS"] = ref_pivot["NS"].fillna(0)
    ref_pivot["Delta_AbsStructCoh"] = np.abs((ref_pivot["SA"] - ref_pivot["NS"]))

    # 2. Pivot data layout for paired metric tracking validation rows
    index_cols = ["NetType", "MAN_code", "BaseNet", "Rep"]
    paired_df = (
        df_10x.pivot(index=index_cols, columns="SelfActivation", values=target_metric)
        .reset_index()
        .dropna(subset=["SA", "NS"])
    )

    for net_type, group_data in paired_df.groupby("NetType"):
        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        # Filter and apply sorting order based on structural difference metrics
        net_ref_pivot = ref_pivot[ref_pivot["NetType"] == net_type].sort_values(
            by=["MAN_code", "Delta_AbsStructCoh"]
        )

        ordered_nets = [
            n
            for n in net_ref_pivot["BaseNet"].tolist()
            if n in group_data["BaseNet"].values
        ]
        if not ordered_nets:
            continue

        # Melt dataset back for Seaborn rendering integration paths
        plot_melt = group_data.melt(
            id_vars=["BaseNet", "MAN_code"],
            value_vars=["NS", "SA"],
            var_name="Condition",
            value_name="MetricValue",
        )

        # Mappingthe NS and SA to Absent and Present
        plot_melt["Condition"] = plot_melt["Condition"].map(
            {"NS": "Absent", "SA": "Present"}
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        # Render background base boxplot frame using your structural layout configuration ordering
        sns.boxplot(
            data=plot_melt,
            x="BaseNet",
            y="MetricValue",
            hue="Condition",
            order=ordered_nets,
            # palette={"NS": nord_colors["green"], "SA": nord_colors["yellow"]},
            palette={"Absent": nord_colors["green"], "Present": nord_colors["yellow"]},
            width=0.6,
            linewidth=1.1,
            # showfliers=False,
            ax=ax,
        )

        # Add headroom margins above the maximum data coordinate for significance symbols
        y_max_data = plot_melt["MetricValue"].max()
        ax.set_ylim(bottom=-y_max_data * 0.03, top=y_max_data * 1.25)

        # -----------------------------------------------------------------
        # STATANNOTATIONS OVERLAY PIPELINE (STAR-RATED FORMAT)
        # -----------------------------------------------------------------
        # annotation_pairs = [((net, "NS"), (net, "SA")) for net in ordered_nets]
        # Update pairs to use the new names
        annotation_pairs = [((net, "Absent"), (net, "Present")) for net in ordered_nets]

        try:
            annotator = Annotator(
                ax=ax,
                pairs=annotation_pairs,
                data=plot_melt,
                x="BaseNet",
                y="MetricValue",
                hue="Condition",
                order=ordered_nets,
            )

            annotator.configure(
                test="t-test_paired",
                text_format="star",
                loc="inside",
                color=nord_colors["gray"],
                line_width=1.2,
                verbose=False,
            )

            annotator.apply_and_annotate()

        except Exception as e:
            print(f"Warning: statannotations execution bypassed for {net_type}: {e}")

        # Add subtle vertical grid separators to visually isolate the unique MAN_code blocks
        man_codes = [b.split("_")[0] for b in ordered_nets]
        for i in range(1, len(man_codes)):
            if man_codes[i] != man_codes[i - 1]:
                ax.axvline(
                    x=i - 0.5,
                    color=nord_colors["gray"],
                    linestyle="--",
                    alpha=0.55,
                    zorder=0,
                )

        # Formatting aesthetics configuration
        # ax.set_xlabel(
        #     r"Network Variants (Grouped by MAN $\rightarrow$ Sorted by $\Delta |C_{\mathrm{struct}}| \,\, [|C_{\mathrm{struct, SA}}| - |C_{\mathrm{struct, NS}}|]$)"
        # )
        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"10x Susceptibility Mapping: Paired SA vs NS Profile | {net_type}", pad=20
        )

        # Replace underscores with hyphens in the x-axis labels
        clean_labels = [net.replace("_", "-") for net in ordered_nets]
        ax.set_xticks(range(len(ordered_nets)))
        ax.set_xticklabels(clean_labels, rotation=90, ha="center")

        # Position external legend frame on the top-right margins
        ax.legend(
            # title="Circuit\nType",
            title="Self-Activation",
            frameon=True,
            facecolor="none",
            edgecolor=nord_colors["gray"],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )

        plt.tight_layout()

        filename = f"Paired_SA_vs_NS_10x_{net_type}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_sa_ns_sfc_bars_10x_F(cae_df, df, target_metric, save_dir, nord_colors):
    """
    Plots a clean, publication-grade bar chart showing the Symmetric Fold Change
    (NS - SA) / (NS + SA) of Means at 10x scale.

    Aesthetics:
        - X-axis labels rotated strictly to 90 degrees for tight alignment.
        - Network names format underscores to hyphens for cleaner reading.
        - Linear Y-axis naturally bounded between [-1, 1].
        - Zoned background shading with a custom top-right legend.
    """
    set_global_nord_style()

    df_10x = cae_df[cae_df["Scale"] == 10].copy()
    if df_10x.empty:
        return

    if "MAN_code" not in df_10x.columns:
        df_10x["MAN_code"] = df_10x["BaseNet"].astype(str).str.split("_").str[0]

    # Clean, ultra-compact Y-axis label reflecting SFC
    # metric_labels = {
    #     "CAE_NumTeams": r"SFC $\frac{\langle\mathrm{NS}\rangle - \langle\mathrm{SA}\rangle}{\langle\mathrm{NS}\rangle + \langle\mathrm{SA}\rangle}$ | $\Delta \mathrm{Teams}$",
    #     "Norm_CAE_NumTeams": r"SFC $\frac{\langle\mathrm{NS}\rangle - \langle\mathrm{SA}\rangle}{\langle\mathrm{NS}\rangle + \langle\mathrm{SA}\rangle}$ | $\mathrm{Norm.\,}\Delta \mathrm{Teams}$",
    # }
    metric_labels = {
        "CAE_NumTeams": r"ND ${|C_{struct}|}_{basal}$",
        "Norm_CAE_NumTeams": r"Normalised Difference",
    }
    clean_ylabel = metric_labels.get(target_metric, "Symmetric Fold Change")

    # 1. Extract reference structural coherence configurations at Density == 1.0 for X-sorting
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["NetType", "SelfActivation", "MAN_code", "BaseNet"])["StructCoh"]
        .mean()
        .reset_index()
    )

    ref_pivot = ref_df.pivot(
        index=["NetType", "MAN_code", "BaseNet"],
        columns="SelfActivation",
        values="StructCoh",
    ).reset_index()

    ref_pivot["SA"] = ref_pivot["SA"].fillna(0)
    ref_pivot["NS"] = ref_pivot["NS"].fillna(0)
    ref_pivot["Delta_AbsStructCoh"] = np.abs((ref_pivot["SA"] - ref_pivot["NS"]))

    # 2. Calculate cohort-wide means for NS and SA
    mean_df = (
        df_10x.groupby(["NetType", "MAN_code", "BaseNet", "SelfActivation"])[
            target_metric
        ]
        .mean()
        .unstack(level="SelfActivation")
        .reset_index()
    )

    # -----------------------------------------------------------------
    # SYMMETRIC FOLD CHANGE (SFC) CALCULATION
    # -----------------------------------------------------------------
    mean_df["SFC"] = (mean_df["NS"] - mean_df["SA"]) / (mean_df["NS"] + mean_df["SA"])
    # Handle mathematical edge cases where both NS and SA are identically 0.0
    mean_df["SFC"] = mean_df["SFC"].fillna(0.0)
    print(mean_df)

    for net_type, group_data in mean_df.groupby("NetType"):
        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        # Apply intentional structural difference sorting order along the X-axis
        net_ref_pivot = ref_pivot[ref_pivot["NetType"] == net_type].sort_values(
            by=["MAN_code", "Delta_AbsStructCoh"]
        )

        ordered_nets = [
            n
            for n in net_ref_pivot["BaseNet"].tolist()
            if n in group_data["BaseNet"].values
        ]
        if not ordered_nets:
            continue

        plot_df = group_data.set_index("BaseNet").reindex(ordered_nets).reset_index()

        # Format the display strings to replace "_" with "-" for cleaner X-axis labels
        plot_df["DisplayNet"] = plot_df["BaseNet"].str.replace("_", "-")

        # -----------------------------------------------------------------
        # CONTINUOUS COLORMAP GENERATION
        # -----------------------------------------------------------------
        cmap_colors = [nord_colors["purple"], nord_colors["orange"]]
        diverging_cmap = mcolors.LinearSegmentedColormap.from_list(
            "Bar_Gradient", cmap_colors
        )

        # Symmetrize boundaries strictly around zero based on the maximum absolute SFC reach
        max_abs_val = plot_df["SFC"].abs().max()
        # Fallback to prevent flatlining if all values are perfectly 0
        if max_abs_val == 0:
            max_abs_val = 1.0

        norm = mcolors.Normalize(vmin=-max_abs_val, vmax=max_abs_val)
        bar_colors = [diverging_cmap(norm(val)) for val in plot_df["SFC"]]

        fig, ax = plt.subplots(figsize=(6, 5))

        # 3. Render compact bar frame natively on the linear scale
        bars = ax.bar(
            x=plot_df["DisplayNet"],
            height=plot_df["SFC"],
            color=bar_colors,
            edgecolor=nord_colors["dark"],
            linewidth=1.1,
            width=0.40,
            zorder=3,
        )

        # Subtle vertical separation lines between unique MAN code blocks
        man_codes = [b.split("_")[0] for b in ordered_nets]
        for i in range(1, len(man_codes)):
            if man_codes[i] != man_codes[i - 1]:
                ax.axvline(
                    x=i - 0.5,
                    color=nord_colors["gray"],
                    linestyle="--",
                    alpha=0.35,
                    zorder=1,
                )

        # Add a crisp structural baseline axis at y=0.0
        ax.axhline(
            y=0.0, color=nord_colors["dark"], linestyle="-", linewidth=1.5, zorder=2
        )

        # Dynamic limits and Headroom setup
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(min(y_min * 1.15, -0.2), max(y_max * 1.15, 0.2))
        y_min_new, y_max_new = ax.get_ylim()

        # -----------------------------------------------------------------
        # ZONED BACKGROUND SHADING & CUSTOM LEGEND
        # -----------------------------------------------------------------
        # Add subtle background spans for visual zoning (zorder=0 keeps them behind bars)
        ax.axhspan(0, y_max_new, color=nord_colors["orange"], alpha=0.20, zorder=0)
        ax.axhspan(y_min_new, 0, color=nord_colors["purple"], alpha=0.20, zorder=0)

        # Create custom patch handles for the legend
        legend_elements = [
            mpatches.Patch(
                facecolor=nord_colors["orange"],
                alpha=0.9,
                edgecolor=nord_colors["dark"],
                linewidth=0.8,
                label="SA Stabilizes",
            ),
            mpatches.Patch(
                facecolor=nord_colors["purple"],
                alpha=0.9,
                edgecolor=nord_colors["dark"],
                linewidth=0.8,
                label="SA Destabilizes",
            ),
        ]

        # Use handlelength/handleheight to shrink the visual box size
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor=nord_colors["gray"],
            framealpha=1.0,
            handlelength=1.0,
            handleheight=1.0,
            fontsize=12,
        )

        # # Labels & Aesthetics
        # ax.set_xlabel(
        #     r"Network Variants (Grouped by MAN $\rightarrow$ Sorted by $\Delta |C_{\mathrm{struct}}| \,\, [|C_{\mathrm{struct, SA}} - C_{\mathrm{struct, NS}}|]$)"
        # )
        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"10x Susceptibility Change: SFC Landscape | Model: {net_type}",
            pad=18,
        )

        # Forced 90-degree rotation and centered horizontal alignment to line up with ticks
        ax.tick_params(axis="x", rotation=90)
        for tick in ax.get_xticklabels():
            tick.set_ha("center")

        plt.tight_layout()

        filename = f"SFC_Means_Bars_10x_{net_type}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_coherence_vs_susceptibility_correlation_10x_F(
    cae_df, df, target_metric, save_dir, nord_colors
):
    """
    Plots the absolute structural coherence differences against the Symmetric Fold Change
    (SFC) of the Means for 10x scale data.

    Calculates and reports separate Spearman correlations for Cyclic, Feedforward,
    and Complete topologies, as well as the Overall correlation, using scientific
    notation for p-values inside a prominently highlighted legend box.
    """
    set_global_nord_style()

    df_10x = cae_df[cae_df["Scale"] == 10].copy()
    if df_10x.empty:
        return

    if "MAN_code" not in df_10x.columns:
        df_10x["MAN_code"] = df_10x["BaseNet"].astype(str).str.split("_").str[0]

    # metric_labels = {
    #     "CAE_NumTeams": r"SFC $\frac{\langle\mathrm{NS}\rangle - \langle\mathrm{SA}\rangle}{\langle\mathrm{NS}\rangle + \langle\mathrm{SA}\rangle}$ | $\Delta \mathrm{Teams}$",
    #     "Norm_CAE_NumTeams": r"SFC $\frac{\langle\mathrm{NS}\rangle - \langle\mathrm{SA}\rangle}{\langle\mathrm{NS}\rangle + \langle\mathrm{SA}\rangle}$ | $\mathrm{Norm.\,}\Delta \mathrm{Teams}$",
    # }
    metric_labels = {
        "CAE_NumTeams": r"ND ${|C_{struct}|}_{basal}$",
        "Norm_CAE_NumTeams": r"Normalised Difference",
        "Norm_CAE_NumPreSplitTeams": r"Normalised Difference",
    }
    clean_ylabel = metric_labels.get(target_metric, "Symmetric Fold Change")

    # 1. Extract reference structural coherence configurations at Density == 1.0
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["NetType", "SelfActivation", "MAN_code", "BaseNet"])["StructCoh"]
        .mean()
        .reset_index()
    )

    ref_pivot = ref_df.pivot(
        index=["NetType", "MAN_code", "BaseNet"],
        columns="SelfActivation",
        values="StructCoh",
    ).reset_index()

    ref_pivot["SA"] = ref_pivot["SA"].fillna(0)
    ref_pivot["NS"] = ref_pivot["NS"].fillna(0)
    ref_pivot["Delta_AbsStructCoh"] = np.abs((ref_pivot["SA"] - ref_pivot["NS"]))

    # 2. Calculate cohort-wide means for NS and SA
    mean_df = (
        df_10x.groupby(["NetType", "MAN_code", "BaseNet", "SelfActivation"])[
            target_metric
        ]
        .mean()
        .unstack(level="SelfActivation")
        .reset_index()
    )

    # -----------------------------------------------------------------
    # SYMMETRIC FOLD CHANGE (SFC) CALCULATION ON MEANS
    # -----------------------------------------------------------------
    mean_df["SFC"] = (mean_df["NS"] - mean_df["SA"]) / (mean_df["NS"] + mean_df["SA"])
    # Handle mathematical edge cases where both NS and SA are identically 0.0
    mean_df["SFC"] = mean_df["SFC"].fillna(0.0)

    # Merge structural predictors with our susceptibility ratio outcomes using dual keys
    correlation_df = pd.merge(
        mean_df,
        ref_pivot[["NetType", "BaseNet", "Delta_AbsStructCoh"]],
        on=["NetType", "BaseNet"],
    )

    # 3. Categorize into Structural Profiles based on BaseNet topology naming
    conditions = [
        correlation_df["BaseNet"].str.startswith("030C"),
        correlation_df["BaseNet"].str.startswith("030T"),
    ]
    choices = ["Cyclic", "Feedforward"]
    correlation_df["Group Profile"] = np.select(conditions, choices, default="Complete")

    for net_type, group_data in correlation_df.groupby("NetType"):
        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))

        # Isolate individual data streams for split scatter rendering
        cyclic_df = group_data[group_data["Group Profile"] == "Cyclic"]
        ff_df = group_data[group_data["Group Profile"] == "Feedforward"]
        comp_df = group_data[group_data["Group Profile"] == "Complete"]

        # --- D. Compute Overall Spearman Statistics (All Categories) ---
        overall_label = "Spearman Correlation"
        if len(group_data) >= 2:
            rho_all, p_all = spearmanr(
                group_data["Delta_AbsStructCoh"], group_data["SFC"]
            )
            overall_label += f"\n$\\rho$ = {rho_all:.2f}, $p$ = {p_all:.2e}"

        # 4. Render Standardized Aggregate Scatter Points
        scatter_comp = ax.scatter(
            comp_df["Delta_AbsStructCoh"],
            comp_df["SFC"],
            facecolor="none",
            edgecolor=nord_colors["red"],
            marker="o",
            linewidth=3.5,
            s=200,
            label="Complete",
            zorder=3,
        )

        scatter_cyc = ax.scatter(
            cyclic_df["Delta_AbsStructCoh"],
            cyclic_df["SFC"],
            facecolor="none",
            edgecolor=nord_colors["yellow"],
            marker="s",
            linewidth=3.5,
            s=200,
            label="Cyclic",
            alpha=0.90,
            zorder=3,
        )

        scatter_ff = ax.scatter(
            ff_df["Delta_AbsStructCoh"],
            ff_df["SFC"],
            facecolors="none",
            edgecolors=nord_colors["green"],
            marker="^",
            linewidth=3.5,
            s=200,
            label="Feedforward",
            alpha=1.0,
            zorder=4,
        )

        # Baseline zero anchor layout line (NS == SA)
        ax.axhline(
            y=0.0,
            color=nord_colors["dark"],
            linestyle="-",
            linewidth=1.5,
            alpha=0.5,
            zorder=1,
        )

        # Formatting Aesthetics
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel(r"$\Delta |C_{\mathrm{struct}}|$")
        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"Structural Delta vs Susceptibility Profile 10x | {net_type}", pad=15
        )

        # First legend box: Topology Groups
        leg1 = ax.legend(
            handles=[scatter_comp, scatter_cyc, scatter_ff],
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor=nord_colors["gray"],
            framealpha=1.0,
            shadow=False,
            fontsize=12,
            markerscale=0.5,
        )
        ax.add_artist(leg1)

        # Second legend box: Overall Correlation
        (dummy_plot,) = ax.plot([], [], linestyle="none", marker="")
        ax.legend(
            handles=[dummy_plot],
            labels=[overall_label],
            loc="upper right",
            bbox_to_anchor=(1.0, 0.75),
            frameon=True,
            facecolor="white",
            edgecolor=nord_colors["gray"],
            framealpha=1.0,
            shadow=False,
            fontsize=12,
            handlelength=0,
            handletextpad=0,
        )

        plt.tight_layout()

        filename = (
            f"Correlation_Coherence_vs_Susceptibility_10x_{net_type}_{target_metric}"
        )
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_cae_by_network_class_F(cae_df, target_metric, save_dir, nord_colors):
    """
    Plots boxplots + stripplots showing the distribution of a CAE metric
    grouped by Network Topology Classification (Complete, Cyclic, Feedforward).
    Includes statannotations to compare the topologies and annotates the
    Mean +/- Std Dev directly above the data clusters.
    """
    from statannotations.Annotator import Annotator

    set_global_nord_style()

    metric_labels = {
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_NumPreSplitTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
    }

    # Create a working copy and drop missing values for the metric
    plot_df = cae_df.dropna(subset=[target_metric]).copy()

    if plot_df.empty:
        print("Warning: Missing data for network class distribution.")
        return

    # Categorize networks into the 3 target structural profiles
    conditions = [
        plot_df["BaseNet"].str.startswith("030C"),
        plot_df["BaseNet"].str.startswith("030T"),
    ]
    choices = ["Cyclic", "Feedforward"]
    plot_df["Network Class"] = np.select(conditions, choices, default="Complete")

    class_order = ["Complete", "Cyclic", "Feedforward"]

    # Assign distinct colors to each class (used for the scatter points)
    class_palette = {
        "Complete": nord_colors["blue"],
        "Cyclic": nord_colors["green"],
        "Feedforward": nord_colors["orange"],
    }

    # Dynamic labels
    # clean_ylabel = f"Cumulative Error\n[{target_metric.replace('_', ' ')}]"
    clean_ylabel = metric_labels[target_metric]

    # Plot loops: One per NetType and SelfActivation
    for (net_type, sa), group_data in plot_df.groupby(["NetType", "SelfActivation"]):
        png_dir = save_dir / "png"
        svg_dir = save_dir / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 6))

        # 1. Stripplot FIRST (zorder=1 so it renders underneath the box frame)
        sns.stripplot(
            data=group_data,
            x="Network Class",
            y=target_metric,
            order=class_order,
            palette=class_palette,
            size=6,
            jitter=True,
            ax=ax,
            zorder=1,
        )

        # Iterate through the generated scatter points to manually adjust face/edge alphas
        for collection in ax.collections:
            colors = collection.get_facecolors()
            if len(colors) > 0:
                # Copy the mapped palette colors for the edges and make them fully opaque
                edge_colors = colors.copy()
                edge_colors[:, 3] = 1.0
                collection.set_edgecolors(edge_colors)
                collection.set_linewidth(1.0)

                # Set the face fills to a very low alpha (alpha=0.25)
                face_colors = colors.copy()
                face_colors[:, 3] = 0.25
                collection.set_facecolors(face_colors)

        # 2. Boxplot SECOND (zorder=2)
        sns.boxplot(
            data=group_data,
            x="Network Class",
            y=target_metric,
            order=class_order,
            palette=class_palette,
            showfliers=False,
            width=0.25,
            ax=ax,
            zorder=2,
        )

        # 3. Strip the box fill to make it completely hollow with a dark frame
        for patch in ax.patches:
            if hasattr(patch, "get_facecolor"):
                patch.set_facecolor("none")  # Completely transparent fill
                patch.set_edgecolor(nord_colors["dark"])  # Solid dark frame
                patch.set_linewidth(2.0)
                patch.set_alpha(1.0)

        # Ensure all whiskers, caps, and median lines are identically thick and dark
        for line in ax.lines:
            line.set_color(nord_colors["dark"])
            line.set_linewidth(2.0)

        # -----------------------------------------------------------------
        # STATANNOTATIONS OVERLAY & MEAN/STD ANNOTATIONS
        # -----------------------------------------------------------------
        y_max = group_data[target_metric].max()

        # Give extra headroom so stat-stars clear the mean text without flying off the chart
        ax.set_ylim(bottom=-y_max * 0.05, top=y_max * 1.10)

        # Compare all three topological classes against each other
        pairs = [
            ("Complete", "Cyclic"),
            ("Cyclic", "Feedforward"),
            ("Complete", "Feedforward"),
        ]

        try:
            annotator = Annotator(
                ax,
                pairs,
                data=group_data,
                x="Network Class",
                y=target_metric,
                order=class_order,
            )
            annotator.configure(
                test="Mann-Whitney",
                text_format="star",
                loc="inside",
            )
            annotator.apply_and_annotate()
        except Exception as e:
            print(
                f"Statannotations skipped for {net_type} {sa} due to variance limits: {e}"
            )

        # Formatting Constraints
        ax.set_xlabel("Network Topology Class", labelpad=10)
        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"Error Distribution by Topology | {net_type} | {sa}",
            pad=20,
            fontsize=15,
        )

        plt.tight_layout()

        filename = f"Dist_NetworkClass_{net_type}_{sa}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)

    print(f"Saved Topology Class Distributions for {target_metric}")


def plot_cae_vs_basal_coherence_scatter_F(
    cae_df, df, target_metric, save_dir, nord_colors
):
    """
    Plots a scatterplot of the target CAE metric vs. the Basal Absolute Structural Coherence.
    Points are colored by the network Scale (Orange, Green, Purple) for high contrast,
    and shapes represent the structural profile with thickened borders.
    Appends Spearman correlation stats inside a top-right text box.
    """
    from scipy.stats import spearmanr

    set_global_nord_style()

    # 1. Extract reference coherence at Density = 1.0 (Basal Coherence)
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["NetType", "SelfActivation", "BaseNet"])["StructCoh"]
        .mean()
        .reset_index()
    )
    ref_df["AbsStructCoh"] = ref_df["StructCoh"].abs()

    # 2. Merge with the CAE dataframe to align the replicate points
    plot_df = pd.merge(
        cae_df, ref_df, on=["NetType", "SelfActivation", "BaseNet"], how="inner"
    )

    if "ScaleStr" not in plot_df.columns:
        plot_df["ScaleStr"] = plot_df["Scale"].astype(str) + "x"

    scale_strs = sorted(
        plot_df["ScaleStr"].unique(), key=lambda x: int(x.replace("x", ""))
    )

    # 3. Define structural profiles for marker shapes
    conditions = [
        plot_df["BaseNet"].str.startswith("030C"),
        plot_df["BaseNet"].str.startswith("030T"),
    ]
    choices = ["Cyclic", "Feedforward"]
    plot_df["Group Profile"] = np.select(conditions, choices, default="Complete")

    # Map shapes: Complete=Circle, Cyclic=Square, Feedforward=Triangle
    profile_markers = {"Complete": "o", "Cyclic": "s", "Feedforward": "^"}

    # Use highly contrasting Nord colors for the scales
    scale_palette = [
        nord_colors["orange"],
        nord_colors["green"],
        nord_colors["purple"],
    ]
    if len(scale_strs) > 3:
        scale_palette = NORD_PALETTE[: len(scale_strs)]

    palette_dict = dict(zip(scale_strs, scale_palette))

    # Y-axis labels mapping
    metric_labels = {
        "CAE_NumTeams": r"Cumulative $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
        "CAE_NumPreSplitTeams": r"Cumulative $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_NumPreSplitTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
    }
    clean_ylabel = metric_labels.get(target_metric, target_metric.replace("_", " "))

    # 4. Plotting Loop
    for (net_type, sa), group_data in plot_df.groupby(["NetType", "SelfActivation"]):
        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6.5, 5))

        # Build the scatterplot with enhanced aesthetics
        sns.scatterplot(
            data=group_data,
            x="AbsStructCoh",
            y=target_metric,
            hue="ScaleStr",
            hue_order=scale_strs,
            palette=palette_dict,
            style="Group Profile",
            markers=profile_markers,
            alpha=0.65,
            s=100,  # Larger marker size
            edgecolor=nord_colors["dark"],  # Crisp dark border
            linewidth=1.3,  # Thicker edges for visibility
            ax=ax,
        )

        # -----------------------------------------------------------------
        # SPEARMAN CORRELATION STATS BOX
        # -----------------------------------------------------------------
        valid_data = group_data[["AbsStructCoh", target_metric]].dropna()
        if len(valid_data) > 2:
            rho, pval = spearmanr(valid_data["AbsStructCoh"], valid_data[target_metric])

            # Format text with scientific notation for p-value
            stat_text = f"Spearman $\\rho$: {rho:.2f}\n$p$-value: {pval:.2e}"

            # Anchor text box inside the plot at the top right
            ax.text(
                0.96,
                0.96,
                stat_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    fc="none",
                ),
            )

        # Formatting and Layout
        ax.set_xlabel(r"${|C_{struct}|}_{basal}$")
        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"{clean_ylabel} vs. Basal Coherence\n{net_type} | {sa}",
            pad=15,
        )

        # Extract the default legend handles and labels generated by Seaborn
        handles, labels = ax.get_legend_handles_labels()

        # Update the specific headers within the labels list
        clean_labels = [
            "Scale"
            if label == "ScaleStr"
            else "Topology\nClass"
            if label == "Group Profile"
            else label
            for label in labels
        ]

        # Move the legend entirely out of the plot and apply the new labels
        ax.legend(
            handles=handles,
            labels=clean_labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
            facecolor="none",
            fontsize=13,
        )

        plt.tight_layout()

        filename = f"Scatter_CAE_vs_BasalCoh_{sa}_{net_type}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


#########################################################################################
### HI Plots
#########################################################################################


def plot_hi_vs_er_cae_distributions_combined_stats_F(
    cae_df, df, target_metric, save_dir, nord_colors
):
    """
    Plots grouped boxplots + stripplots comparing HI vs ER CAE distributions across ALL scales.
    Generates one plot per SelfActivation state (SA, NS).
    X-axis is grouped by MAN Code and sorted by basal absolute structural coherence.
    Hue separates the NetType (HI vs ER).
    Includes statannotations to compare HI vs ER significance per motif.
    Stripplot is placed behind the boxplot, and X-axis labels include the structural coherence.
    """
    set_global_nord_style()

    metric_labels = {
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_NumPreSplitTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
    }

    # 1. Filter out 030T networks
    filtered_cae = cae_df[~cae_df["BaseNet"].str.startswith("030T")].copy()

    hi_cae = filtered_cae[filtered_cae["NetType"] == "HI"].copy()
    er_cae = filtered_cae[filtered_cae["NetType"] == "ER"].copy()

    if hi_cae.empty or er_cae.empty:
        print("Warning: Missing either HI or ER data.")
        return

    # 2. Determine which Coherence metric to use for sorting
    coh_metric = "PreSplitStructCoh" if "PreSplit" in target_metric else "StructCoh"

    # 3. Extract reference coherence at Density = 1.0 (Using HI as the structural baseline)
    ref_df = (
        df[
            (df["Density"] == 1.0)
            & (df["NetType"] == "HI")
            & (~df["BaseNet"].str.startswith("030T"))
        ]
        .groupby(["SelfActivation", "BaseNet"])[coh_metric]
        .mean()
        .reset_index()
    )
    ref_df["AbsStructCoh"] = ref_df[coh_metric]
    ref_df["MAN_code"] = ref_df["BaseNet"].astype(str).str.split("_").str[0]

    # Combine data streams using the EXACT same target_metric for both HI and ER
    hi_sub = hi_cae[["BaseNet", "NetType", "SelfActivation", target_metric]].copy()
    er_sub = er_cae[["BaseNet", "NetType", "SelfActivation", target_metric]].copy()

    plot_df = pd.concat([hi_sub, er_sub], ignore_index=True)
    plot_df = plot_df.dropna(subset=[target_metric])

    # Dynamic labels
    # clean_ylabel = f"Cumulative Error (All Scales)\n[{target_metric.replace('_', ' ')}]"
    clean_ylabel = metric_labels[target_metric]
    # if "PreSplit" in target_metric:
    #     x_label = r"Network Variants (Grouped by MAN $\rightarrow$ Sorted by HI $|C_{\mathrm{struct, Pre-Split}}|$ Baseline)"
    # else:
    #     x_label = r"Network Variants (Grouped by MAN $\rightarrow$ Sorted by HI $|C_{\mathrm{struct}}|$ Baseline)"

    net_palette = {"HI": nord_colors["orange"], "ER": nord_colors["purple"]}

    # 4. Plot loops: One per SelfActivation (Aggregating all scales)
    for sa, group_data in plot_df.groupby("SelfActivation"):
        # png_dir = save_dir / "png"
        # svg_dir = save_dir / "svg"
        png_dir = save_dir / "HI" / "png"
        svg_dir = save_dir / "HI" / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        sa_refs = ref_df[ref_df["SelfActivation"] == sa].sort_values(
            by=["MAN_code", "AbsStructCoh"]
        )
        ordered_nets = [
            n for n in sa_refs["BaseNet"].tolist() if n in group_data["BaseNet"].values
        ]

        if not ordered_nets:
            continue

        # Create mapping dictionary for appending coherence to labels
        sa_net_to_coh = dict(zip(sa_refs["BaseNet"], sa_refs["AbsStructCoh"]))

        # Widen figure slightly to accommodate the annotations
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        # Plot Stripplot FIRST so it renders behind the boxplot (zorder=1)
        sns.stripplot(
            data=group_data,
            x="BaseNet",
            y=target_metric,
            hue="NetType",
            hue_order=["HI", "ER"],
            order=ordered_nets,
            dodge=True,
            palette=[nord_colors["dark"], nord_colors["dark"]],
            alpha=0.35,
            size=4,
            jitter=True,
            ax=ax,
            legend=False,
            zorder=1,
        )

        # Plot Boxplot SECOND (zorder=2)
        sns.boxplot(
            data=group_data,
            x="BaseNet",
            y=target_metric,
            hue="NetType",
            hue_order=["HI", "ER"],
            order=ordered_nets,
            palette=net_palette,
            showfliers=False,
            linewidth=1.2,
            ax=ax,
            zorder=2,
        )

        # -----------------------------------------------------------------
        # STATANNOTATIONS OVERLAY
        # -----------------------------------------------------------------
        pairs = [((net, "HI"), (net, "ER")) for net in ordered_nets]

        # Give extra headroom for the stat-stars
        y_max = group_data[target_metric].max()
        ax.set_ylim(bottom=-y_max * 0.05, top=y_max * 1.3)

        try:
            annotator = Annotator(
                ax,
                pairs,
                data=group_data,
                x="BaseNet",
                y=target_metric,
                hue="NetType",
                order=ordered_nets,
                hue_order=["HI", "ER"],
            )
            annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
            annotator.apply_and_annotate()
        except Exception as e:
            print(f"Statannotations skipped for {sa} due to variance limits: {e}")

        # Add Demarcation Grid Lines
        man_codes = [b.split("_")[0] for b in ordered_nets]
        for i in range(1, len(man_codes)):
            if man_codes[i] != man_codes[i - 1]:
                ax.axvline(
                    x=i - 0.5,
                    color=nord_colors["gray"],
                    linestyle="--",
                    alpha=0.35,
                    zorder=0,
                )

        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"Cumulative Error (HI vs ER | All Scales Combined) | {sa}",
            pad=20,
            fontsize=16,
        )

        # Append absolute structural coherence to x-axis labels
        new_labels = [
            f"({sa_net_to_coh.get(net, 0.0):.2f}) {net.replace('_', '-')}"
            # f"{net.replace('_', '-')}"
            for net in ordered_nets
        ]
        ax.set_xticks(range(len(ordered_nets)))
        ax.set_xticklabels(new_labels, rotation=90, ha="center")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles[:2],
            labels=labels[:2],
            title="Network\nType",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=True,
            # facecolor="#f2f4f8",
            edgecolor=nord_colors["gray"],
        )

        plt.tight_layout()

        filename = f"Dist_HI_vs_ER_CombinedStats_{sa}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_hi_cae_distributions_all_scales_F(
    cae_df, df, target_metric, save_dir, nord_colors
):
    """
    Plots boxplots + stripplots showing HI CAE distributions across ALL scales.
    Generates one plot per SelfActivation state (SA, NS).
    X-axis is grouped by MAN Code and sorted by raw basal structural coherence.
    Box colors are mapped to the RAW basal structural coherence (-1 to 1)
    using a continuous Red -> Purple -> Blue colormap.
    Includes statistical annotations for adjacent networks within the same MAN code.
    """

    set_global_nord_style()

    metric_labels = {
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_NumPreSplitTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
    }

    # 1. Filter out 030T networks and keep ONLY HI networks
    filtered_cae = cae_df[~cae_df["BaseNet"].str.startswith("030T")].copy()
    hi_cae = filtered_cae[filtered_cae["NetType"] == "HI"].copy()

    if hi_cae.empty:
        print("Warning: Missing HI data.")
        return

    # 2. Determine which Coherence metric to use for sorting and coloring
    coh_metric = "PreSplitStructCoh" if "PreSplit" in target_metric else "StructCoh"

    # 3. Extract reference coherence at Density = 1.0 (Using HI as the structural baseline)
    ref_df = (
        df[
            (df["Density"] == 1.0)
            & (df["NetType"] == "HI")
            & (~df["BaseNet"].str.startswith("030T"))
        ]
        .groupby(["SelfActivation", "BaseNet"])[coh_metric]
        .mean()
        .reset_index()
    )
    # Keeping raw value as per your instructions (no .abs()) for both sorting and coloring
    ref_df["AbsStructCoh"] = ref_df[coh_metric]
    ref_df["MAN_code"] = ref_df["BaseNet"].astype(str).str.split("_").str[0]

    plot_df = hi_cae.dropna(subset=[target_metric])

    # Dynamic labels
    # clean_ylabel = f"Cumulative Error (All Scales)\n[{target_metric.replace('_', ' ')}]"
    clean_ylabel = metric_labels[target_metric]
    # if "PreSplit" in target_metric:
    #     x_label = r"Network Variants (Grouped by MAN $\rightarrow$ Sorted by $|C_{\mathrm{struct, Pre-Split}}|$ Baseline)"
    # else:
    #     x_label = r"Network Variants (Grouped by MAN $\rightarrow$ Sorted by $|C_{\mathrm{struct}}|$ Baseline)"

    # Create the continuous Red -> Purple -> Blue Colormap (No White)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "NordRedWhiteBlue",
        [nord_colors["red"], "#eceff4", nord_colors["blue"]],
    )
    # Lock the scale strictly from -1.0 to 1.0
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    # 4. Plot loops: One per SelfActivation (Aggregating all scales)
    for sa, group_data in plot_df.groupby("SelfActivation"):
        # png_dir = save_dir / "png"
        # svg_dir = save_dir / "svg"
        png_dir = save_dir / "HI" / "png"
        svg_dir = save_dir / "HI" / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        # Sort by Raw Coherence
        sa_refs = ref_df[ref_df["SelfActivation"] == sa].sort_values(
            by=["MAN_code", "AbsStructCoh"]
        )
        ordered_nets = [
            n for n in sa_refs["BaseNet"].tolist() if n in group_data["BaseNet"].values
        ]

        if not ordered_nets:
            continue

        # Build color dictionary mapping each BaseNet to its exact raw coherence color
        palette_dict = {
            net: cmap(norm(sa_refs[sa_refs["BaseNet"] == net][coh_metric].values[0]))
            for net in ordered_nets
        }

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot Stripplot FIRST so it renders behind the boxplot (zorder=1)
        sns.stripplot(
            data=group_data,
            x="BaseNet",
            y=target_metric,
            order=ordered_nets,
            # color=nord_colors["dark"],  # Uniform dark dots
            palette=palette_dict,
            alpha=0.15,
            size=6,
            jitter=True,
            ax=ax,
            zorder=1,
        )

        # Plot Boxplot SECOND (zorder=2) mapped to the continuous palette
        sns.boxplot(
            data=group_data,
            x="BaseNet",
            y=target_metric,
            order=ordered_nets,
            palette=palette_dict,
            showfliers=False,
            linewidth=1.2,
            saturation=1.0,  # Ensures exact Nord hex matches are rendered
            ax=ax,
            width=0.4,
            zorder=2,
        )

        # -----------------------------------------------------------------
        # STATANNOTATIONS OVERLAY (Within-Group Adjacent Pairs)
        # -----------------------------------------------------------------
        pairs = []
        man_codes_list = [b.split("_")[0] for b in ordered_nets]

        # Build pairs strictly between adjacent networks that share the same MAN_code
        for i in range(len(ordered_nets) - 1):
            if man_codes_list[i] == man_codes_list[i + 1]:
                pairs.append((ordered_nets[i], ordered_nets[i + 1]))

        # Give extra headroom to accommodate the statistical brackets
        y_max = group_data[target_metric].max()
        ax.set_ylim(bottom=-y_max * 0.05, top=y_max * 1.3)

        if pairs:
            try:
                annotator = Annotator(
                    ax,
                    pairs,
                    data=group_data,
                    x="BaseNet",
                    y=target_metric,
                    order=ordered_nets,
                )
                annotator.configure(
                    test="Mann-Whitney", text_format="star", loc="inside"
                )
                annotator.apply_and_annotate()
            except Exception as e:
                print(f"Statannotations skipped for {sa} due to variance limits: {e}")

        # Add Demarcation Grid Lines
        for i in range(1, len(man_codes_list)):
            if man_codes_list[i] != man_codes_list[i - 1]:
                ax.axvline(
                    x=i - 0.5,
                    color=nord_colors["gray"],
                    linestyle="--",
                    alpha=0.35,
                    zorder=0,
                )

        # ax.set_xlabel(x_label)
        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"Cumulative Error Colored by Basal Coherence (HI | All Scales) | {sa}",
            pad=20,
        )

        # Clean X-axis labels (replace underscores with hyphens, NO numerical values appended)
        clean_labels = [net.replace("_", "-") for net in ordered_nets]
        ax.set_xticks(range(len(ordered_nets)))
        ax.set_xticklabels(clean_labels, rotation=90, ha="center")

        # Reserve right side of the figure for the colorbar
        plt.tight_layout(rect=[0, 0, 0.88, 1])

        # Render the Continuous Colorbar
        cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(r"${C_{struct}}$", rotation=270, labelpad=20)
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

        filename = f"Dist_HI_Coherence_ColorMap_{sa}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_ns_vs_sa_boxplots_10x_with_stats_F(
    cae_df, df, target_metric, save_dir, nord_colors
):
    """
    Plots 10x scale paired boxes comparing SA and NS side-by-side.

    Sorting: Networks are grouped by MAN_code, and within each family block,
             sorted sequentially by the difference in Absolute Structural Coherence
             at Density = 1.0 between the SA and NS versions (|C_struct,SA| - |C_struct,NS|).
    Significance: Default statannotations star system (*, **, ***, ns) inside the grid headroom.
    Legend: Bounded external axis frame on the top right.
    """
    set_global_nord_style()

    df_10x = cae_df[cae_df["Scale"] == 10].copy()
    if df_10x.empty:
        return

    # 1. Filter out 030T networks AND restrict to 10x scale
    df_10x = df_10x[
        (~df_10x["BaseNet"].str.startswith("030T")) & (df_10x["Scale"] == 10)
    ].copy()

    if "MAN_code" not in df_10x.columns:
        df_10x["MAN_code"] = df_10x["BaseNet"].astype(str).str.split("_").str[0]

    metric_labels = {
        "CAE_NumTeams": r"Cumulative $| \Delta \mathrm{Teams} |$",
        "Norm_CAE_NumTeams": r"Cumulative Norm. $| \Delta \mathrm{Teams} |$",
    }
    clean_ylabel = metric_labels.get(target_metric, target_metric.replace("_", " "))

    # 1. Extract reference background structural coherence configurations at Density == 1.0
    ref_df = (
        df[df["Density"] == 1.0]
        .groupby(["NetType", "SelfActivation", "MAN_code", "BaseNet"])["StructCoh"]
        .mean()
        .reset_index()
    )

    # Pivot ref_df to compute the precise difference in absolute structural coherence between conditions
    ref_pivot = ref_df.pivot(
        index=["NetType", "MAN_code", "BaseNet"],
        columns="SelfActivation",
        values="StructCoh",
    ).reset_index()

    ref_pivot["SA"] = ref_pivot["SA"].fillna(0)
    ref_pivot["NS"] = ref_pivot["NS"].fillna(0)
    ref_pivot["Delta_AbsStructCoh"] = np.abs((ref_pivot["SA"] - ref_pivot["NS"]))

    # 2. Pivot data layout for paired metric tracking validation rows
    index_cols = ["NetType", "MAN_code", "BaseNet", "Rep"]
    paired_df = (
        df_10x.pivot(index=index_cols, columns="SelfActivation", values=target_metric)
        .reset_index()
        .dropna(subset=["SA", "NS"])
    )

    for net_type, group_data in paired_df.groupby("NetType"):
        png_dir = save_dir / net_type / "png"
        svg_dir = save_dir / net_type / "svg"
        png_dir.mkdir(parents=True, exist_ok=True)
        svg_dir.mkdir(parents=True, exist_ok=True)

        # Filter and apply sorting order based on structural difference metrics
        net_ref_pivot = ref_pivot[ref_pivot["NetType"] == net_type].sort_values(
            by=["MAN_code", "Delta_AbsStructCoh"]
        )

        ordered_nets = [
            n
            for n in net_ref_pivot["BaseNet"].tolist()
            if n in group_data["BaseNet"].values
        ]
        if not ordered_nets:
            continue

        # Melt dataset back for Seaborn rendering integration paths
        plot_melt = group_data.melt(
            id_vars=["BaseNet", "MAN_code"],
            value_vars=["NS", "SA"],
            var_name="Condition",
            value_name="MetricValue",
        )

        # Mappingthe NS and SA to Absent and Present
        plot_melt["Condition"] = plot_melt["Condition"].map(
            {"NS": "Absent", "SA": "Present"}
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        # Render background base boxplot frame using your structural layout configuration ordering
        sns.boxplot(
            data=plot_melt,
            x="BaseNet",
            y="MetricValue",
            hue="Condition",
            order=ordered_nets,
            # palette={"NS": nord_colors["green"], "SA": nord_colors["yellow"]},
            palette={"Absent": nord_colors["green"], "Present": nord_colors["yellow"]},
            width=0.6,
            linewidth=1.1,
            # showfliers=False,
            ax=ax,
        )

        # Add headroom margins above the maximum data coordinate for significance symbols
        y_max_data = plot_melt["MetricValue"].max()
        ax.set_ylim(bottom=-y_max_data * 0.03, top=y_max_data * 1.25)

        # -----------------------------------------------------------------
        # STATANNOTATIONS OVERLAY PIPELINE (STAR-RATED FORMAT)
        # -----------------------------------------------------------------
        # annotation_pairs = [((net, "NS"), (net, "SA")) for net in ordered_nets]
        # Update pairs to use the new names
        annotation_pairs = [((net, "Absent"), (net, "Present")) for net in ordered_nets]

        try:
            annotator = Annotator(
                ax=ax,
                pairs=annotation_pairs,
                data=plot_melt,
                x="BaseNet",
                y="MetricValue",
                hue="Condition",
                order=ordered_nets,
            )

            annotator.configure(
                test="t-test_paired",
                text_format="star",
                loc="inside",
                color=nord_colors["gray"],
                line_width=1.2,
                verbose=False,
            )

            annotator.apply_and_annotate()

        except Exception as e:
            print(f"Warning: statannotations execution bypassed for {net_type}: {e}")

        # Add subtle vertical grid separators to visually isolate the unique MAN_code blocks
        man_codes = [b.split("_")[0] for b in ordered_nets]
        for i in range(1, len(man_codes)):
            if man_codes[i] != man_codes[i - 1]:
                ax.axvline(
                    x=i - 0.5,
                    color=nord_colors["gray"],
                    linestyle="--",
                    alpha=0.55,
                    zorder=0,
                )

        # Formatting aesthetics configuration
        # ax.set_xlabel(
        #     r"Network Variants (Grouped by MAN $\rightarrow$ Sorted by $\Delta |C_{\mathrm{struct}}| \,\, [|C_{\mathrm{struct, SA}}| - |C_{\mathrm{struct, NS}}|]$)"
        # )
        ax.set_ylabel(clean_ylabel)
        ax.set_title(
            f"10x Susceptibility Mapping: Paired SA vs NS Profile | {net_type}", pad=20
        )

        # Replace underscores with hyphens in the x-axis labels
        clean_labels = [net.replace("_", "-") for net in ordered_nets]
        ax.set_xticks(range(len(ordered_nets)))
        ax.set_xticklabels(clean_labels, rotation=90, ha="center")

        # Position external legend frame on the top-right margins
        ax.legend(
            # title="Circuit\nType",
            title="Self-Activation",
            frameon=True,
            facecolor="none",
            edgecolor=nord_colors["gray"],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )

        plt.tight_layout()

        filename = f"Dist_NS_vs_SA_Boxplots_10x_{net_type}_{target_metric}"
        fig.savefig(
            png_dir / f"{filename}.png",
            format="png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig.savefig(
            svg_dir / f"{filename}.svg",
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


if __name__ == "__main__":
    # data_path = Path("./ScaledCohResults_FCM/CompiledScaledSummary_FCM.parquet")
    data_path = Path("./ScaledCohResults/CompiledScaledSummary.parquet")

    preprocess_data_path = data_path.parent / "ProcessedSummary.parquet"

    # Running pre-processing only if the file does not exist
    if not preprocess_data_path.exists():
        print("Running pre-processing")
        cohres_df = preprocess_data(data_path)
        # Saving the preprocesed data to save time
        cohres_df.to_parquet(preprocess_data_path)
    else:
        print("Skipping pre-processing. File exists.")
        # Reading the preprocessed data if the file exists
        cohres_df = pd.read_parquet(preprocess_data_path)

    print(cohres_df.dtypes)
    print(cohres_df.shape)

    cohres_df = cohres_df[cohres_df["BaseNet"] != "300_NNNPPN"]

    cae_data_path = data_path.parent / "CAEData.parquet"

    # Running the cae calcualtion only if not done earlier
    if not cae_data_path.exists():
        print("Running CAE calcualtion")
        cae_df = calculate_cae_metrics(cohres_df)
        cae_df.to_parquet(data_path.parent / "CAEData.parquet")
    else:
        print("Skipping CAE calcualtion. File exists.")
        cae_df = pd.read_parquet(cae_data_path)

    print(cae_df.columns)
    print(cae_df.dtypes)
    print(cae_df.shape)

    # Setting up the folder to save plots
    plot_save_dir = Path("./AE_Plots/F2")
    plot_save_dir.mkdir(exist_ok=True, parents=True)

    metrics_to_plot = ["AE_NumTeams", "AE_StructCoh"]

    for metric in metrics_to_plot:
        plot_metric_trajectories_panel_F(cohres_df, metric, plot_save_dir, NORD_COLORS)

    # Setting up the folder to save plots
    plot_save_dir = Path("./AE_Plots/F3")
    plot_save_dir.mkdir(exist_ok=True)
    scale_metrics = [
        "CAE_NumTeams",
    ]
    for metric in scale_metrics:
        plot_scale_trajectories_with_stats_F(
            cae_df=cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
    scale_metrics = [
        "Norm_CAE_NumTeams",
    ]
    for metric in scale_metrics:
        plot_cae_vs_basal_coherence_scatter_F(
            cae_df=cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )

    # Setting up the folder to save plots
    plot_save_dir = Path("./AE_Plots/F4")
    plot_save_dir.mkdir(exist_ok=True)

    metrics_for_topology = [
        "Norm_CAE_NumTeams",
    ]

    for metric in metrics_for_topology:
        plot_cae_by_network_class_F(
            cae_df=cae_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
        plot_ratio_heatmaps_F(
            cae_df=cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )

    # Setting up the folder to save plots
    plot_save_dir = Path("./AE_Plots/F5")
    plot_save_dir.mkdir(exist_ok=True)

    metrics_for_topology = [
        "Norm_CAE_NumTeams",
    ]

    for metric in metrics_for_topology:
        plot_sa_vs_ns_paired_10x_F(
            cae_df=cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
        plot_sa_ns_sfc_bars_10x_F(
            cae_df=cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
        plot_coherence_vs_susceptibility_correlation_10x_F(
            cae_df=cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )

    # Setting up the folder to save plots
    plot_save_dir = Path("./AE_Plots/F9")
    plot_save_dir.mkdir(exist_ok=True)

    dist_comp_metrics = [
        "Norm_CAE_NumPreSplitTeams",
    ]

    hi_only_cae_df = cae_df[cae_df["NetType"] == "HI"].copy()
    hi_only_cohres_df = cohres_df[cohres_df["NetType"] == "HI"].copy()

    for metric in dist_comp_metrics:
        plot_hi_cae_distributions_all_scales_F(
            cae_df=hi_only_cae_df,
            df=hi_only_cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
        plot_hi_vs_er_cae_distributions_combined_stats_F(
            cae_df=cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
        plot_ns_vs_sa_boxplots_10x_with_stats_F(
            cae_df=hi_only_cae_df,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )

        # Removing the feedorward networks from the HI-ER comparision analysis
        cae_df_no_ff = cae_df[~cae_df["BaseNet"].str.startswith("030T")].copy()
        cohres_df_no_ff = cohres_df[~cohres_df["BaseNet"].str.startswith("030T")].copy()
        plot_coherence_vs_susceptibility_correlation_10x_F(
            cae_df=cae_df_no_ff,
            df=cohres_df,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
        plot_cae_vs_basal_coherence_scatter_F(
            cae_df=cae_df_no_ff,
            df=cohres_df_no_ff,
            target_metric=metric,
            save_dir=plot_save_dir,
            nord_colors=NORD_COLORS,
        )
