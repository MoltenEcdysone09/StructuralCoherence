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


def analyze_and_plot_rnap_F(data_dir, regdb_dir, plot_dir):
    """
    Preprocesses RNAP expression data against RegulonDB target metrics
    and generates an lmplot of Absolute In Coherence vs Absolute LFC.
    """
    plot_dir.mkdir(exist_ok=True, parents=True)

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

        # Check if this is time data set or not
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
            expdf = pd.merge(
                expdf, reg_df, left_on="gname", right_on="Node", how="inner"
            )
            print(expdf)
            print(expdf.columns)

        else:
            continue

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

    hue_order = sorted(plot_long["Condition"].unique())

    new_labels = []
    for cond in hue_order:
        subset = plot_long[plot_long["Condition"] == cond]
        # CLEANING: Drop rows where x or y is NaN or Infinite just for the math
        valid_data = (
            subset[["MeanAbsInCoh", "AbsLFC"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
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
            # label = f"{cond.replace('_', ' ').replace('X', 'x')}\n($r={r_value:.2f}, p-value={p_value:.2f})"
            label = (
                rf"{cond.replace('_', ' ').replace('X', 'x')}"
                + "\n"
                + rf"($r={r_value:.2f}, p\text{{-value}}={p_value:.2f}$)"
            )
        else:
            label = f"{cond} (N/A)"
        new_labels.append(label)

    fig_w, fig_h = 10, 6

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

    ax.set_xlabel("|Incoming Coherence|")
    ax.set_ylabel(r"|$\log_{10}$(Fold Change)|")

    handles, _ = ax.get_legend_handles_labels()

    if len(handles) > len(hue_order):
        handles = handles[: len(hue_order)]

    # Plot the legend on the right side, stacked vertically
    plt.legend(
        handles=handles,
        labels=new_labels,
        title="Condition",
        bbox_to_anchor=(1.05, 1.0),  # Anchors legend to the right of the plot
        loc="upper left",  # Sets the origin of the legend box
        ncol=1,  # Stacks items vertically (one below the other)
        borderaxespad=0,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(
        plot_dir / "AbsInCoh_AbsLFC_Corr.png",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.savefig(
        plot_dir / "AbsInCoh_AbsLFC_Corr.svg", bbox_inches="tight", transparent=True
    )
    plt.close()


if __name__ == "__main__":
    # Specify the directories
    base_data_dir = Path("./lf2c_rnap/")
    output_plot_dir = Path("./iModulon_Plots/Fig11/")
    regulondb_dir = Path(
        "./AbasyCohResults_Targeted/511145_v2022_sRDB22_eStrong_regNetwork_Strong/"
    )

    # Run the analysis and plotting
    analyze_and_plot_rnap_F(
        data_dir=base_data_dir, regdb_dir=regulondb_dir, plot_dir=output_plot_dir
    )
