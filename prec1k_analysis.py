import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path

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


# =====================================================================
# 2. DATA PREPROCESSING MODULE
# =====================================================================


def convert_trn_to_topo(trn_path, output_dir, output_filename="ecoli_trn.topo"):
    """Converts a TRN CSV file into a Source-Target-Type Topology file."""
    df = pd.read_csv(trn_path)
    subset = df[["regulator_name", "target_name", "effect"]].copy()
    subset = subset.dropna(subset=["regulator_name", "target_name"])

    effect_map = {"activation": 1, "repression": -1}
    subset["Type"] = subset["effect"].str.lower().map(effect_map)
    subset = subset.dropna(subset=["Type"])
    subset["Type"] = subset["Type"].astype(int)

    topo_df = subset.rename(
        columns={"regulator_name": "Source", "target_name": "Target"}
    )[["Source", "Target", "Type"]]

    output_path = os.path.join(output_dir, output_filename)
    topo_df.to_csv(output_path, sep=" ", index=False)
    return topo_df


def load_and_preprocess_data(data_dir, cmat_parquet_path):
    """
    Loads raw matrices and executes expression filter pipelines.
    Returns structured data frames and the filtered active module list.
    """
    # Load expression expression datasets
    M = pd.read_csv(os.path.join(data_dir, "M.csv"), index_col=0)
    thresholds = pd.read_csv(
        os.path.join(data_dir, "e_coli_modulome_thresholds.csv"),
        index_col="imodulon_name",
    )
    trn = pd.read_csv(os.path.join(data_dir, "e_coli_modulome_trn.csv"))
    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
    imodulon_table = pd.read_csv(
        os.path.join(data_dir, "imodulon_table.csv"), index_col=0
    )
    A = pd.read_csv(os.path.join(data_dir, "A.csv"), index_col=0)

    # Map baseline gene identifiers
    gene_map = (
        trn[["target_id", "target_name"]]
        .drop_duplicates("target_id")
        .set_index("target_id")["target_name"]
        .to_dict()
    )

    # Generate network topography formats
    convert_trn_to_topo(os.path.join(data_dir, "e_coli_modulome_trn.csv"), data_dir)

    # Load experimental interaction matrix layout
    cmat = pd.read_parquet(cmat_parquet_path)

    # Execute step-wise sample constraints filtering
    wt_mask = metadata["Strain"].str.contains("MG1655", case=False, na=False)
    is_evolved = metadata["Evolved Sample"].astype(str).str.lower() == "true"
    selected_samples = metadata[wt_mask & ~is_evolved].index
    selected_samples = [s for s in selected_samples if s in A.columns]
    print(f"Selected Samples: {len(selected_samples)} (WT/Non-Evolved)")

    # Execute functional module layout filtering
    if "Category" in imodulon_table.columns:
        non_genomic = imodulon_table[imodulon_table["Category"] != "Genomic"].index
        candidate_modules = [m for m in non_genomic if m in M.columns]
    else:
        candidate_modules = M.columns.tolist()

    candidate_modules = [m for m in candidate_modules if m in A.index]

    # Calculate variance boundaries across valid experimental dimensions
    A_subset = A.loc[candidate_modules, selected_samples]
    active_modules = A_subset[A_subset.std(axis=1) > 0.1].index.tolist()
    print(f"Active Modules: {len(active_modules)} (Filtered from {len(M.columns)})")

    return M, cmat, thresholds, gene_map, active_modules


# =====================================================================
# 3. CORE COHERENCE ANALYSIS MODULE
# =====================================================================


def get_imodulon_genes(imodulon_name, M, thresholds, gene_map):
    """Retrieves significant weight distributions for a specified target iModulon."""
    if imodulon_name not in M.columns:
        return f"Error: '{imodulon_name}' not found in M matrix columns."

    weights = M[imodulon_name]
    thresh = (
        thresholds.loc[imodulon_name, "threshold"]
        if imodulon_name in thresholds.index
        else 0.05
    )
    significant_genes = weights[weights.abs() > thresh].to_frame(name="Weight")

    significant_genes["Gene_Name"] = [
        gene_map.get(str(g), str(g)) for g in significant_genes.index
    ]
    return significant_genes.sort_values(by="Weight", ascending=False)


def analyze_module_coherence(target_mod, M, cmat, thresholds, gene_map, verbose=False):
    """Performs pair-wise comparison and classification tests against interaction targets."""
    if verbose:
        print(f"Processing: {target_mod}...", end=" ")

    if target_mod not in M.columns:
        if verbose:
            print("Error: Not in M matrix.")
        return None

    thresh = (
        thresholds.loc[target_mod, "threshold"]
        if target_mod in thresholds.index
        else 0.05
    )
    weights = M[target_mod]
    df_genes = weights[weights.abs() > thresh].to_frame(name="Weight")

    df_genes["Gene_Name"] = [gene_map.get(str(g), str(g)) for g in df_genes.index]
    b_numbers = df_genes.index.to_series()
    df_genes["Gene_Name"] = df_genes["Gene_Name"].fillna(b_numbers)

    module_gene_list = df_genes["Gene_Name"].unique()
    valid_genes = [g for g in module_gene_list if g in cmat.index and g in cmat.columns]

    if len(valid_genes) < 2:
        if verbose:
            print("Skipped (Not enough genes).")
        return None

    module_cmat = cmat.loc[valid_genes, valid_genes]
    pairs_df = module_cmat.stack().reset_index()
    pairs_df.columns = ["Gene_A", "Gene_B", "CohVal"]

    # Explicitly purge unmeasured array spacing to guarantee correct sparse coordinates
    pairs_df = pairs_df.dropna(subset=["CohVal"])
    pairs_df = pairs_df[pairs_df["Gene_A"] != pairs_df["Gene_B"]].copy()

    if pairs_df.empty:
        if verbose:
            print("Skipped (No interactions found).")
        return None

    pairs_df["Gene_A"] = pairs_df["Gene_A"].astype(str)
    pairs_df["Gene_B"] = pairs_df["Gene_B"].astype(str)

    weight_dict = df_genes.set_index("Gene_Name")["Weight"].to_dict()
    pairs_df["Weight_A"] = pairs_df["Gene_A"].map(weight_dict)
    pairs_df["Weight_B"] = pairs_df["Gene_B"].map(weight_dict)
    pairs_df["Weight_Product"] = pairs_df["Weight_A"] * pairs_df["Weight_B"]

    if pairs_df["Weight_Product"].std() == 0 or pairs_df["CohVal"].std() == 0:
        rho, p_val = 0, 1.0
    else:
        rho, p_val = stats.spearmanr(pairs_df["Weight_Product"], pairs_df["CohVal"])

    pairs_df["Predicted_Pos"] = pairs_df["Weight_Product"] > 0
    pairs_df["Actual_Pos"] = pairs_df["CohVal"] > 0

    report_dict = classification_report(
        pairs_df["Actual_Pos"],
        pairs_df["Predicted_Pos"],
        output_dict=True,
        zero_division=0,
    )

    if verbose:
        print(f"Done. (Pairs: {len(pairs_df)})")

    return {
        "Module": target_mod,
        "Gene_Count": len(df_genes),
        "Valid_Genes_In_Cmat": len(valid_genes),
        "Interacting_Pairs": len(pairs_df),
        "Spearman_Rho": rho,
        "P_Value": p_val,
        "Precision": report_dict.get("True", {}).get("precision", 0),
        "Recall": report_dict.get("True", {}).get("recall", 0),
        "F1_Score": report_dict.get("True", {}).get("f1-score", 0),
    }


def analyze_module_coherence_undirected(
    target_mod, M, cmat, thresholds, gene_map, verbose=False
):
    """
    Analyzes Module vs Interaction Network using an upper triangular matrix strategy.

    Directly compares C_ij and C_ji for each distinct pair of genes to categorize
    them as cooperative (True) or antagonistic (False):
    - Both directions are NaN: Removed from comparison.
    - One direction is NaN, the other is negative: Antagonistic.
    - One direction is NaN, the other is positive: Cooperative.
    - Signs mismatch between directions: Antagonistic.
    - Signs match between directions: Cooperative.
    """
    if verbose:
        print(f"Processing: {target_mod}...", end=" ")

    if target_mod not in M.columns:
        if verbose:
            print("Error: Not in M matrix.")
        return None

    thresh = (
        thresholds.loc[target_mod, "threshold"]
        if target_mod in thresholds.index
        else 0.05
    )
    weights = M[target_mod]
    df_genes = weights[weights.abs() > thresh].to_frame(name="Weight")

    df_genes["Gene_Name"] = [gene_map.get(str(g), str(g)) for g in df_genes.index]
    b_numbers = df_genes.index.to_series()
    df_genes["Gene_Name"] = df_genes["Gene_Name"].fillna(b_numbers)

    module_gene_list = df_genes["Gene_Name"].unique()
    valid_genes = [g for g in module_gene_list if g in cmat.index and g in cmat.columns]

    if len(valid_genes) < 2:
        if verbose:
            print("Skipped (Not enough genes).")
        return None

    # 1. Subset the square 2D matrix slice for valid genes
    module_cmat = cmat.loc[valid_genes, valid_genes]
    mat_values = module_cmat.to_numpy(dtype=float)
    n = len(valid_genes)

    # 2. Get coordinates for the upper triangular matrix (k=1 excludes the diagonal)
    row_idx, col_idx = np.triu_indices(n, k=1)

    # 3. Pull C_ij and C_ji simultaneously using vectorized lookups
    v1 = mat_values[row_idx, col_idx]  # Forward direction: Gene_1 -> Gene_2
    v2 = mat_values[col_idx, row_idx]  # Reverse direction: Gene_2 -> Gene_1

    # 4. Construct the pair framework directly from the vector arrays
    undirected_df = pd.DataFrame(
        {
            "Gene_1": [valid_genes[r] for r in row_idx],
            "Gene_2": [valid_genes[c] for c in col_idx],
            "v1": v1,
            "v2": v2,
        }
    )

    # Strategy Rule: If both directional links are unmeasured (NaN), drop the pair
    undirected_df = undirected_df.dropna(subset=["v1", "v2"], how="all").copy()

    if undirected_df.empty:
        if verbose:
            print("Skipped (No valid undirected pairs after NaN filtering).")
        return None

    # Strategy Rule: Determine ground truth sign consensus
    def determine_actual_coherence(row):
        val1, val2 = row["v1"], row["v2"]
        if np.isnan(val1):
            return val2 > 0  # Cooperative if positive, antagonistic if negative
        if np.isnan(val2):
            return val1 > 0  # Cooperative if positive, antagonistic if negative

        # Signs match -> Cooperative (True), Signs mismatch -> Antagonistic (False)
        return (val1 > 0) == (val2 > 0)

    undirected_df["Actual_Pos"] = undirected_df.apply(
        determine_actual_coherence, axis=1
    )

    # Map module weights to the gene pairs
    weight_dict = df_genes.set_index("Gene_Name")["Weight"].to_dict()
    undirected_df["Weight_1"] = undirected_df["Gene_1"].map(weight_dict)
    undirected_df["Weight_2"] = undirected_df["Gene_2"].map(weight_dict)
    undirected_df["Weight_Product"] = (
        undirected_df["Weight_1"] * undirected_df["Weight_2"]
    )

    # Formulate binary predictions based on weight product sign
    undirected_df["Predicted_Pos"] = undirected_df["Weight_Product"] > 0

    # Create composite representative values for rank correlation
    undirected_df["CohVal_Rep"] = undirected_df["v1"].fillna(undirected_df["v2"])
    both_measured = undirected_df["v1"].notna() & undirected_df["v2"].notna()
    undirected_df.loc[both_measured, "CohVal_Rep"] = (
        undirected_df["v1"] + undirected_df["v2"]
    ) / 2

    # Calculate statistics
    if (
        undirected_df["Weight_Product"].std() == 0
        or undirected_df["CohVal_Rep"].std() == 0
    ):
        rho, p_val = 0, 1.0
    else:
        rho, p_val = stats.spearmanr(
            undirected_df["Weight_Product"], undirected_df["CohVal_Rep"]
        )

    report_dict = classification_report(
        undirected_df["Actual_Pos"],
        undirected_df["Predicted_Pos"],
        output_dict=True,
        zero_division=0,
    )

    if verbose:
        print(f"Done. (Undirected Pairs: {len(undirected_df)})")

    return {
        "Module": target_mod,
        "Gene_Count": len(df_genes),
        "Valid_Genes_In_Cmat": len(valid_genes),
        "Interacting_Pairs": len(undirected_df),
        "Spearman_Rho": rho,
        "P_Value": p_val,
        "Precision": report_dict.get("True", {}).get("precision", 0),
        "Recall": report_dict.get("True", {}).get("recall", 0),
        "F1_Score": report_dict.get("True", {}).get("f1-score", 0),
    }


def run_precision_recall_analysis(
    active_modules, M, cmat, thresholds, gene_map, verbose=False
):
    """Loops through active modules to execute batch processing metrics generation."""
    print(f"Starting analysis on {len(active_modules)} modules...")
    results_list = []
    for mod in active_modules:
        # res = analyze_module_coherence(
        #     mod, M, cmat, thresholds, gene_map, verbose=verbose
        # )
        res = analyze_module_coherence_undirected(
            mod, M, cmat, thresholds, gene_map, verbose=verbose
        )
        if res:
            results_list.append(res)

    results_df = pd.DataFrame(results_list)
    print(f"\nAnalysis Complete. Successfully analyzed {len(results_df)} modules.")
    return results_df


# =====================================================================
# 4. VISUALIZATION AND REPORTING MODULE
# =====================================================================


def plot_coverage_regression_F(plot_df, save_dir):
    """Generates the Network Coverage scaling regression layout plot."""
    plt.figure(figsize=(6, 6))
    r, p = stats.spearmanr(plot_df["Gene_Count"], plot_df["Valid_Genes_In_Cmat"])

    sns.regplot(
        data=plot_df,
        x="Gene_Count",
        y="Valid_Genes_In_Cmat",
        color=NORD_COLORS["green"],
        scatter_kws={"s": 120},
        line_kws={"color": NORD_COLORS["red"], "alpha": 0.9},
    )

    max_val = (
        min(plot_df["Gene_Count"].max(), plot_df["Valid_Genes_In_Cmat"].max()) + 30
    )
    plt.plot(
        [0, max_val],
        [0, max_val],
        ls="--",
        c=NORD_COLORS["dark"],
        linewidth=2.5,
        alpha=0.7,
        label="Perfect Coverage",
    )

    stats_text = f"Spearman ρ = {r:.2f}\np-value = {p:.2e}"
    plt.text(
        0.80,
        0.08,
        stats_text,
        transform=plt.gca().transAxes,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round", fc="white", alpha=0.9, edgecolor=NORD_COLORS["gray"]
        ),
        # fontsize=10,
        color=NORD_COLORS["dark"],
    )

    plt.xlabel("Genes in Module")
    plt.ylabel("Genes from Coherence Matrix")
    # plt.title("Network Coverage")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "corr_cohvsmodulon.png"), dpi=300, transparent=True
    )
    plt.savefig(
        os.path.join(save_dir, "corr_cohvsmodulon.svg"), dpi=300, transparent=True
    )
    plt.close()


def plot_performance_quadrants_F(plot_df, save_dir):
    """Generates the discrete performance classification quadrant overview scatter plot."""
    plt.figure(figsize=(6, 5))
    nord_f1_cmap = LinearSegmentedColormap.from_list(
        "nord_f1_seq",
        [NORD_COLORS["blue"], NORD_COLORS["green"]],
    )

    sns.scatterplot(
        data=plot_df,
        x="Precision",
        y="Recall",
        size="F1_Score",
        hue="F1_Score",
        sizes=(20, 200),
        palette=nord_f1_cmap,
        color=NORD_COLORS["blue"],
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.2,
        alpha=0.9,
        legend=False,
    )

    plt.axhline(0.5, color=NORD_COLORS["gray"], linestyle="--", linewidth=1.5)
    plt.axvline(0.5, color=NORD_COLORS["gray"], linestyle="--", linewidth=1.5)

    q1 = len(plot_df[(plot_df["Precision"] > 0.5) & (plot_df["Recall"] > 0.5)])
    q2 = len(plot_df[(plot_df["Precision"] <= 0.5) & (plot_df["Recall"] > 0.5)])
    q3 = len(plot_df[(plot_df["Precision"] <= 0.5) & (plot_df["Recall"] <= 0.5)])
    q4 = len(plot_df[(plot_df["Precision"] > 0.5) & (plot_df["Recall"] <= 0.5)])

    plt.text(
        0.59,
        0.59,
        f"n={q1}",
        ha="center",
        va="center",
        # fontsize=12,
        color=NORD_COLORS["green"],
    )
    plt.text(
        0.59,
        0.41,
        f"n={q4}",
        ha="center",
        va="center",
        # fontsize=12,
        color=NORD_COLORS["blue"],
    )
    plt.text(
        0.41,
        0.41,
        f"n={q3}",
        ha="center",
        va="center",
        # fontsize=12,
        color=NORD_COLORS["red"],
    )
    plt.text(
        0.41,
        0.59,
        f"n={q2}",
        ha="center",
        va="center",
        # fontsize=12,
        color=NORD_COLORS["orange"],
    )

    norm = Normalize(vmin=plot_df["F1_Score"].min(), vmax=plot_df["F1_Score"].max())
    sm = ScalarMappable(cmap=nord_f1_cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("F1 Score", labelpad=10)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title("Module Performance Quadrants")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "PrecisionvsRecall.png"), dpi=300, transparent=True
    )
    plt.savefig(
        os.path.join(save_dir, "PrecisionvsRecall.svg"), dpi=300, transparent=True
    )
    plt.close()


def plot_volcano_significance_F(plot_df, save_dir):
    """Generates rank correlation volcano plot mapping rho coefficients against significance logs."""
    import numpy as np

    # Create a working copy to avoid SettingWithCopy warnings
    plot_df = plot_df.copy()

    # Define significance thresholds based on your existing axes lines
    p_threshold = -np.log10(0.05)
    rho_threshold = 0.35

    # Categorize points based on correlation direction and significance thresholds
    conditions = [
        (plot_df["-log10(p_value)"] > p_threshold)
        & (plot_df["Spearman_Rho"] > rho_threshold),
        (plot_df["-log10(p_value)"] > p_threshold)
        & (plot_df["Spearman_Rho"] < -rho_threshold),
    ]
    choices = ["Significant Positive", "Significant Negative"]
    plot_df["Significance_Category"] = np.select(
        conditions, choices, default="Not Significant"
    )

    # Map categories to the requested Nord colors
    volcano_palette = {
        "Significant Positive": NORD_COLORS["green"],
        "Significant Negative": NORD_COLORS["red"],
        "Not Significant": NORD_COLORS["gray"],
    }

    plt.figure(figsize=(6, 5))

    sns.scatterplot(
        data=plot_df,
        x="Spearman_Rho",
        y="-log10(p_value)",
        hue="Significance_Category",
        palette=volcano_palette,
        s=120,
        edgecolor=NORD_COLORS["dark"],
        linewidth=1.2,
        alpha=0.8,
        legend=False,
    )

    plt.axhline(
        p_threshold,
        color=NORD_COLORS["gray"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
    )
    plt.axvline(
        rho_threshold,
        color=NORD_COLORS["gray"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
    )
    plt.axvline(
        -rho_threshold,
        color=NORD_COLORS["gray"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
    )

    top_neg_corr = (
        plot_df[plot_df["Spearman_Rho"] < 0]
        .sort_values("Spearman_Rho", ascending=True)
        .head(10)
    )

    if not top_neg_corr.empty:
        y_coords = np.linspace(
            plot_df["-log10(p_value)"].max() * 0.8,
            plot_df["-log10(p_value)"].max() * 0.2,
            len(top_neg_corr),
        )
        for y, (_, row) in zip(y_coords, top_neg_corr.iterrows()):
            # Check if this specific module passed the significance threshold
            is_significant = row["Significance_Category"] != "Not Significant"

            plt.text(
                -0.4,
                y,
                f"{row['Module']}",
                fontsize=9,
                color=NORD_COLORS["red"]
                if is_significant
                else NORD_COLORS["dark"],  # Applies red conditionally
                ha="right",
                va="center",
            )

    plt.xlabel(r"Spearman $\rho$")
    plt.ylabel(r"Significance ($-\log_{10}(p\text{-value})$)")
    plt.tight_layout()

    import os

    plt.savefig(
        os.path.join(save_dir, "Volcano_Spearman.png"), dpi=300, transparent=True
    )
    plt.savefig(
        os.path.join(save_dir, "Volcano_Spearman.svg"), dpi=300, transparent=True
    )
    plt.close()


def generate_plots_F(df, save_dir):
    """Unified master interface executing separated subset graphic pipelines."""
    plot_df = df[df["Interacting_Pairs"] >= 5].copy()
    if "-log10(p_value)" not in plot_df.columns:
        plot_df["-log10(p_value)"] = -np.log10(plot_df["P_Value"] + 1e-300)

    plot_coverage_regression_F(plot_df, save_dir)
    plot_performance_quadrants_F(plot_df, save_dir)
    plot_volcano_significance_F(plot_df, save_dir)


def print_summary_reports_F(results_df):
    """Outputs structured metric tables and discrete partition counters to console logs."""
    top_matches = results_df.sort_values(by="F1_Score", ascending=False)
    print("\n--- Top Modules by Consistency (F1 Score) ---")
    print(
        top_matches[["Module", "Interacting_Pairs", "F1_Score", "Spearman_Rho"]].head(
            10
        )
    )

    top_corr = results_df.sort_values(by="Spearman_Rho", ascending=False)
    print("\n--- Top Modules by Correlation (Spearman Rho) ---")
    print(
        top_corr[
            ["Module", "Interacting_Pairs", "F1_Score", "Spearman_Rho", "P_Value"]
        ].head(10)
    )

    robust_df = results_df[results_df["Interacting_Pairs"] >= 5].copy()
    q1_count = len(
        robust_df[(robust_df["Precision"] > 0.5) & (robust_df["Recall"] > 0.5)]
    )
    q2_count = len(
        robust_df[(robust_df["Precision"] <= 0.5) & (robust_df["Recall"] > 0.5)]
    )
    q3_count = len(
        robust_df[(robust_df["Precision"] <= 0.5) & (robust_df["Recall"] <= 0.5)]
    )
    q4_count = len(
        robust_df[(robust_df["Precision"] > 0.5) & (robust_df["Recall"] <= 0.5)]
    )

    print("\n========================================================")
    print("MODULE PERFORMANCE QUADRANT SUMMARY (Min 5 Interacting Pairs)")
    print("========================================================")
    print(f"Total Robust Modules Evaluated: {len(robust_df)}")
    print(f"Q1 (Good Precision, Good Recall)         : {q1_count} modules")
    print(f"Q2 (Imprecise, High Recall)              : {q2_count} modules")
    print(f"Q3 (Low Precision, Low Recall / Mixed)   : {q3_count} modules")
    print(f"Q4 (Precise, Low Recall)                 : {q4_count} modules")
    print("========================================================")


# =====================================================================
# 5. UNIFIED MAIN PROGRAM FLOW
# =====================================================================

if __name__ == "__main__":
    # Define primary directory variables
    target_data_directory = "precise1k-v1.0"
    coherence_matrix_path = "./AbasyCohResults_Targeted/511145_v2022_sRDB22_eStrong_regNetwork_Strong/511145_v2022_sRDB22_eStrong_regNetwork_Strong_CohMat.parquet"

    # Step 1: Execute clean, logical structural separation mapping inputs
    M, cmat, thresholds, gene_map, active_modules = load_and_preprocess_data(
        data_dir=target_data_directory, cmat_parquet_path=coherence_matrix_path
    )

    # Step 2: Run bulk verification calculations pipeline
    results_df = run_precision_recall_analysis(
        active_modules=active_modules,
        M=M,
        cmat=cmat,
        thresholds=thresholds,
        gene_map=gene_map,
        verbose=False,
    )
    res = analyze_module_coherence_undirected(
        "IS1", M, cmat, thresholds, gene_map, verbose=False
    )

    # # Step 3: Emit numerical analytics data layout maps
    # print_summary_reports(results_df)

    combined_plot_dir = Path("./iModulon_Plots/Fig10")
    combined_plot_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Draw and export decoupled thematic plotting arrays
    generate_plots_F(df=results_df, save_dir=combined_plot_dir)
