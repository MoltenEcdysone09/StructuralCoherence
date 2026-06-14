import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

# --- Organism Name Mapping ---
NETWORK_TO_ORGANISM = {
    "196627_v2020_s21_regNetwork_Strong": "Corynebacterium glutamicum",
    "83332_v2018_s15-16_regNetwork": "Mycobacterium tuberculosis",
    "224308_v2022_sSW22_regNetwork": "Bacillus subtilis",
    "511145_v2022_sRDB22_eStrong_regNetwork_Strong": "Escherichia coli",
    "208964_v2020_sRPA20_regNetwork_Strong": "Pseudomonas aeruginosa",
    "100226_v2019_sA22-DBSCR15_eStrong_regNetwork": "Streptomyces coelicolor",
}

# --- Set font and plot styles ---
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


def plot_combined_outlier_distribution_F(
    networks_data, plot_dir, metric="NumPreSplitGroups", normalize=True
):
    """
    Plots combined KDEs and WT lines for multiple networks on a single plot.
    If normalize=True, uses Min-Max Normalization to make scales comparable.
    If normalize=False, uses raw numbers.
    Adds a systematic horizontal jitter to the WT lines to prevent perfect overlap.
    Uses the native legend placed inside the plot with a semi-transparent background.
    """

    # 1. Enforce the global Nord styling before drawing
    set_global_nord_style()

    # Slightly narrower figure since the legend is moving inside
    fig, ax = plt.subplots(figsize=(6.5, 5))

    color_cycle = itertools.cycle(NORD_PALETTE)
    legend_handles = []

    total_nets = len(networks_data)
    # Define a base offset step (0.008 for normalized 0-1 scale, adjust if needed)
    offset_step = 0.008 if normalize else 0.8

    for idx, net in enumerate(networks_data):
        org_name = net["name"]
        metric_data = net["shuffled_data"]
        wt_val = net["wt_val"]

        color = next(color_cycle)

        # --- DATA PREPARATION (TOGGLEABLE NORMALIZATION) ---
        if normalize:
            # Include BOTH the null distribution and the WT value to find the true min/max
            min_val = min(metric_data.min(), wt_val)
            max_val = max(metric_data.max(), wt_val)

            if (max_val - min_val) == 0:
                plot_data = metric_data - min_val
                plot_wt = wt_val - min_val
            else:
                plot_data = (metric_data - min_val) / (max_val - min_val)
                plot_wt = (wt_val - min_val) / (max_val - min_val)
        else:
            plot_data = metric_data
            plot_wt = wt_val

        # 3. Plot the Null Distribution (Shuffled Networks)
        sns.histplot(
            plot_data,
            element="step",
            fill=True,
            alpha=0.25,
            color=color,
            edgecolor=color,
            linewidth=2.5,
            stat="probability",
            ax=ax,
            zorder=2,
        )

        # Calculate a systematic jitter to separate overlapping lines
        jitter = (idx - (total_nets - 1) / 2) * offset_step

        # 4. Add the WT Vertical Line with the calculated jitter
        ax.axvline(
            plot_wt + jitter,
            color=color,
            linestyle="--",
            linewidth=2.5,
            zorder=3,
        )

        # 5. Calculate Empirical P-value in Scientific Notation (Using RAW data)
        n_extreme = (metric_data >= wt_val).sum()
        total = len(metric_data)
        p_val = n_extreme / total

        if p_val == 0:
            p_val_str = f"< {1 / total:.2e}"
        else:
            p_val_str = f"= {p_val:.2e}"

        # 6. Create the legend entry
        org_italic = f"$\\mathit{{{org_name.replace(' ', r'\ ')}}}$"
        label_text = f"{org_italic}\nEmpirical $p$-value {p_val_str}"

        handle = mpatches.Patch(
            facecolor=color,
            alpha=1.0,  # Opaque colors in the legend marker
            edgecolor=color,
            linewidth=2.5,
            label=label_text,
        )
        legend_handles.append(handle)

    # --- DYNAMIC X-AXIS LABELING ---
    if metric == "NumPreSplitGroups":
        x_label = (
            "Min-Max Normalized Number of Teams" if normalize else "Number of Teams"
        )
    elif metric == "NumGroups":
        x_label = (
            "Min-Max Normalized Number of Teams (WCC)\n(Scaled to Null & WT Range)"
            if normalize
            else "Number of Teams (WCC)"
        )
    else:
        x_label = f"Normalized {metric}" if normalize else f"{metric}"

    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel("Probability", labelpad=10)

    # 8. Position the Native Legend INSIDE the plot
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.92, 0.99),  # Safely tucked in the top right corner
        frameon=True,
        facecolor="white",  # White box background
        framealpha=0.45,  # Semi-transparent so lines underneath are vaguely visible
        edgecolor=NORD_COLORS["gray"],
        fontsize=10,  # Scaled down to fit nicely inside
    )

    plt.tight_layout()

    # Save the combined plot with a dynamic filename
    plot_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "Normalized" if normalize else "Raw"
    file_path = plot_dir / f"Combined_{file_prefix}_{metric}_Hist.png"

    fig.savefig(file_path, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(
        file_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def plot_cohmat_distributions_F(wt_base_dir, shuffle_base_dir, plot_dir):
    """
    Reads the CohMat.parquet files for the WT and Random networks.
    Generates a 2-panel plot:
      1. Overlapping Step Histograms (Density).
      2. Density Difference Plot (WT - Random) to explicitly highlight enrichment zones.
    """
    set_global_nord_style()

    for grn, org_name in NETWORK_TO_ORGANISM.items():
        print(f"\nProcessing CohMat distributions for {org_name}...")

        # ---------------------------------------------------------
        # 1. Extract Data
        # ---------------------------------------------------------
        wt_path = wt_base_dir / grn / f"{grn}_CohMat.parquet"
        if not wt_path.exists():
            continue

        wt_df = pd.read_parquet(wt_path)
        wt_vals = wt_df.to_numpy().flatten()
        wt_vals = wt_vals[~np.isnan(wt_vals)]

        random_vals_list = []
        shuffled_mats_dir = shuffle_base_dir / grn / "Shuffled_CohMats"

        for i in range(1, 51):
            rand_name = f"{grn}_Random{i:03d}"
            rand_path = shuffled_mats_dir / rand_name / f"{rand_name}_CohMat.parquet"
            if rand_path.exists():
                rand_df = pd.read_parquet(rand_path)
                r_vals = rand_df.to_numpy().flatten()
                random_vals_list.append(r_vals[~np.isnan(r_vals)])

        if not random_vals_list:
            continue

        rand_vals_combined = np.concatenate(random_vals_list)

        # ---------------------------------------------------------
        # 2. Calculate Density Differences Mathematically
        # ---------------------------------------------------------
        # Define shared bins so we can subtract the distributions
        n_bins = 50
        bins = np.linspace(-1.0, 1.0, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        # Calculate probability density for both
        wt_density, _ = np.histogram(wt_vals, bins=bins, density=True)
        rand_density, _ = np.histogram(rand_vals_combined, bins=bins, density=True)

        # Calculate Difference (WT - Random)
        density_diff = wt_density - rand_density

        # ---------------------------------------------------------
        # 3. Render 2-Panel Plot
        # ---------------------------------------------------------
        fig, (ax_main, ax_diff) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(6, 5),
            sharex=True,
            gridspec_kw={
                "height_ratios": [2.5, 1.25],
                "hspace": 0.05,
            },  # Tightly stack panels
        )

        # --- TOP PANEL: Overlapping Densities ---
        sns.histplot(
            rand_vals_combined,
            bins=bins,
            element="step",
            fill=True,
            stat="density",
            color=NORD_COLORS["yellow"],
            edgecolor=NORD_COLORS["yellow"],
            alpha=0.20,
            linewidth=2.0,
            label="Random Networks",
            ax=ax_main,
            zorder=1,
        )

        sns.histplot(
            wt_vals,
            bins=bins,
            element="step",
            fill=True,
            stat="density",
            color=NORD_COLORS["green"],
            edgecolor=NORD_COLORS["green"],
            alpha=0.35,
            linewidth=2.5,
            label="WT Network",
            ax=ax_main,
            zorder=2,
        )

        ax_main.set_ylabel("Density", labelpad=10)
        ax_main.set_title(f"{org_name}", pad=15)
        ax_main.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax_main.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

        ax_main.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=2,
            frameon=True,
            edgecolor=NORD_COLORS["gray"],
            fontsize=12,
        )

        # --- BOTTOM PANEL: Density Difference ---
        # Color dynamically: Blue if WT is enriched, Orange if Random is enriched
        diff_colors = [
            NORD_COLORS["green"] if val > 0 else NORD_COLORS["yellow"]
            for val in density_diff
        ]

        ax_diff.bar(
            bin_centers,
            density_diff,
            width=bin_width,
            color=diff_colors,
            edgecolor=NORD_COLORS["dark"],
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )

        # Baseline at 0.0
        ax_diff.axhline(0, color=NORD_COLORS["dark"], linewidth=1.5, zorder=3)

        ax_diff.set_xlabel(r"Coherence Value ($C_{ij}$)", labelpad=10)
        # ax_diff.set_ylabel(r"$\Delta$ Density\n(WT - Random)", fontsize=11)
        ax_diff.set_ylabel(r"$\Delta$ Density" + "\n" + "(WT - Random)")
        ax_diff.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax_diff.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

        # Force symmetry on the difference Y-axis so 0 is exactly in the middle
        max_diff = np.max(np.abs(density_diff)) * 1.15
        if max_diff > 0:
            ax_diff.set_ylim(-max_diff, max_diff)

        plt.tight_layout()

        # Save Systems
        filename = f"CohMat_WT_vs_Random_{grn}"
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

        print(f"  Saved plot to {plot_dir.name}/{filename}.png")


def plot_nodelevel_connectivity_density_heatmaps_F(wt_base_dir, plot_dir):
    """
    Classifies nodes into Input, Middle, and Output based on the WT CohMat.
    Calculates the connection density (Actual Non-NaN Edges / Possible Edges)
    between each source-target level pair.
    Plots a 2x3 panel of 3x3 heatmaps for the 6 organisms without a colorbar.
    Values are annotated in scientific notation up to 3 decimal places.
    Missing structural links are mapped to the Nord dark color natively via the colormap.
    """
    import matplotlib.colors as mcolors

    set_global_nord_style()

    print("\nProcessing NodeLevel Connectivity Density Heatmaps...")

    level_names = ["Input", "Middle", "Output"]

    # 1. Create a 2x3 grid for the 6 organisms
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
    axes = axes.flatten()

    # Create a clean continuous Nord Colormap for the densities
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "NordDensity",
        [NORD_COLORS["orange"], NORD_COLORS["yellow"]],
    )

    # Map NaN values natively to the Nord dark color
    cmap.set_bad(color=NORD_COLORS["dark"])

    for idx, (grn, org_name) in enumerate(NETWORK_TO_ORGANISM.items()):
        wt_path = wt_base_dir / grn / f"{grn}_CohMat.parquet"
        ax = axes[idx]

        if not wt_path.exists():
            ax.set_visible(False)
            continue

        wt_df = pd.read_parquet(wt_path)
        M_abs = np.abs(wt_df.to_numpy())
        valid_mask = ~np.isnan(M_abs)

        # 2. Topologically classify the nodes
        has_out = valid_mask.any(axis=1)
        has_in = valid_mask.any(axis=0)

        masks = {
            "Input": has_out & ~has_in,
            "Middle": has_in & has_out,
            "Output": has_in & ~has_out,
        }

        # 3. Calculate 3x3 Density Matrix
        density_mat = np.zeros((3, 3))

        for i, src in enumerate(level_names):
            for j, tgt in enumerate(level_names):
                src_m = masks[src]
                tgt_m = masks[tgt]

                possible_edges = np.sum(src_m) * np.sum(tgt_m)
                if possible_edges > 0:
                    actual_edges = np.sum(valid_mask[np.ix_(src_m, tgt_m)])
                    density_mat[i, j] = actual_edges / possible_edges
                else:
                    # Explicit NaN for blocks that structurally cannot exist
                    density_mat[i, j] = np.nan

        density_df = pd.DataFrame(density_mat, index=level_names, columns=level_names)

        # Replace the 0s with NaNs so the colormap's set_bad() targets them
        density_df = density_df.replace(0, np.nan)

        # 4. Render the Heatmap
        sns.heatmap(
            data=density_df,
            ax=ax,
            cmap=cmap,
            cbar=False,
            annot=True,
            fmt=".1e",
            linewidths=1.5,
            linecolor="white",
            annot_kws={"fontsize": 13},
        )

        # 5. Panel Formatting
        ax.set_title(org_name, pad=15)

        # Only add outside axis labels to keep the interior clean
        ax.set_ylabel("Source Level" if idx % 3 == 0 else "", labelpad=10)
        ax.set_xlabel("Target Level" if idx >= 3 else "", labelpad=10)

        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    # Main Figure Title
    fig.suptitle("Inter/Intra-Level Structural Connection Density", y=1.02)

    plt.tight_layout()

    # 6. Save outputs
    filename = "CohMat_Global_NodeLevel_Density_Heatmaps"
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


if __name__ == "__main__":
    SHUFFLE_RESULT_DIR = Path("./WTvsShuffledAnalysis_AbasyNets_Targeted/")

    # Specifying the WT result folder
    wt_cohres_df = pd.read_parquet(
        "./AbasyCohResults_Targeted/CompiledTargetSummary.parquet"
    )

    metrics_to_plot = ["NumPreSplitGroups"]
    combined_plot_dir = Path("./GRN_Plots/Fig6")
    combined_plot_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics_to_plot:
        print(f"\nGathering data for metric: {metric}")
        networks_data = []

        for grn in wt_cohres_df["TopoName"].unique():
            rn_path = (
                SHUFFLE_RESULT_DIR
                / grn
                / "Shuffled_CohMats"
                / f"CompiledShuffledSummary_{grn}.parquet"
            )

            if not rn_path.exists():
                print(f"  Warning: Missing shuffled data at {rn_path}. Skipping.")
                continue

            rn_cohres_df = pd.read_parquet(rn_path)

            metric_data = pd.to_numeric(rn_cohres_df[metric])
            wt_value = float(
                wt_cohres_df.loc[wt_cohres_df["TopoName"] == grn, metric].values[0]
            )

            organism_name = NETWORK_TO_ORGANISM.get(grn, grn)

            networks_data.append(
                {
                    "name": organism_name,
                    "shuffled_data": metric_data,
                    "wt_val": wt_value,
                }
            )

        if networks_data:
            plot_combined_outlier_distribution_F(
                networks_data, combined_plot_dir, metric
            )
            print(
                f"Saved combined plot to {combined_plot_dir / f'Combined_Normalized_{metric}_KDE.png'}"
            )

    # =====================================================================
    # Plot CohMat WT vs Random Distributions
    # =====================================================================

    WT_RESULT_DIR = Path("./AbasyCohResults_Targeted")

    # Execute the new CohMat distribution mapping
    plot_cohmat_distributions_F(
        wt_base_dir=WT_RESULT_DIR,
        shuffle_base_dir=SHUFFLE_RESULT_DIR,
        plot_dir=combined_plot_dir,
    )

    plot_nodelevel_connectivity_density_heatmaps_F(
        wt_base_dir=WT_RESULT_DIR,
        plot_dir=combined_plot_dir,
    )
