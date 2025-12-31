import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy import stats
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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


def convert_trn_to_topo(trn_path, output_dir, output_filename="ecoli_trn.topo"):
    """
    Converts a TRN CSV file into a Source-Target-Type Topology file.

    Mapping:
    - Activation -> 1
    - Repression -> 0
    - Unknown/Empty -> Dropped (to keep the network signed and clean)
    """

    # 1. Load the TRN
    print(f"Reading TRN from: {trn_path}")
    df = pd.read_csv(trn_path)

    # Filter only strong
    # df = df[df["evidence"] == 1]

    # 2. Extract relevant columns
    # We use names instead of IDs for readability
    subset = df[["regulator_name", "target_name", "effect"]].copy()

    # 3. Clean Missing Names
    # Drop rows where regulator or target names are missing (NaN)
    subset = subset.dropna(subset=["regulator_name", "target_name"])

    # 4. Map the Effects to 1 and 0
    effect_map = {
        "activation": 1,
        "repression": -1,
        # 'unknown' and nan are excluded automatically
    }

    subset["Type"] = subset["effect"].str.lower().map(effect_map)

    # 5. Filter for Valid Effects
    # We drop rows where Type is NaN (this removes 'unknown' or empty effects)
    subset = subset.dropna(subset=["Type"])
    subset["Type"] = subset["Type"].astype(int)  # Convert float to int

    # 6. Rename Columns to Standard Topology Format
    topo_df = subset.rename(
        columns={"regulator_name": "Source", "target_name": "Target"}
    )[["Source", "Target", "Type"]]

    print(topo_df["Type"].value_counts())

    # 7. Save as TSV with .topo extension
    output_path = os.path.join(output_dir, output_filename)
    topo_df.to_csv(output_path, sep=" ", index=False)
    return topo_df


# 1. Setup Paths
data_dir = "precise1k-v1.0"

# 2. Load Matrices
M = pd.read_csv(os.path.join(data_dir, "M.csv"), index_col=0)

# We use index_col="imodulon_name" so we can look up thresholds by name
thresholds = pd.read_csv(
    os.path.join(data_dir, "e_coli_modulome_thresholds.csv"), index_col="imodulon_name"
)

# 3. Create Gene Map from TRN
trn = pd.read_csv(os.path.join(data_dir, "e_coli_modulome_trn.csv"))
gene_map = (
    trn[["target_id", "target_name"]]
    .drop_duplicates("target_id")
    .set_index("target_id")["target_name"]
    .to_dict()
)

# Converting the trn to topo
convert_trn_to_topo(os.path.join(data_dir, "e_coli_modulome_trn.csv"), data_dir)

# Loading up the coherence matrix
cmat = (
    pd.read_parquet(
        # "./precise1k-v1.0/iModulonCohMatResults/ecoli_trn_sw/ecoli_trn_sw_CohMat.parquet"
        "./AbasyCohResults/511145_v2022_sRDB22_eStrong_regNetwork_Strong/511145_v2022_sRDB22_eStrong_regNetwork_Strong_CohMat.parquet"
    )
    .droplevel(0)
    .droplevel(0, axis=1)
)
# print(cmat)


# ---------------------------------------------------------
# FUNCTION: Get genes for a specific iModulon
# ---------------------------------------------------------
def get_imodulon_genes(imodulon_name):
    if imodulon_name not in M.columns:
        return f"Error: '{imodulon_name}' not found in M matrix columns."

    # 1. Get weights
    weights = M[imodulon_name]

    # 2. Get Threshold
    if imodulon_name in thresholds.index:
        # Access the 'threshold' column for this index
        thresh = thresholds.loc[imodulon_name, "threshold"]
    else:
        print(
            f"Warning: '{imodulon_name}' not found in threshold file. Using default 0.05."
        )
        thresh = 0.05

    # 3. Filter (Absolute value > threshold)
    significant_genes = weights[weights.abs() > thresh].to_frame(name="Weight")

    # 4. Map Names
    significant_genes["Gene_Name"] = significant_genes.index.map(gene_map)

    # 5. Handle Missing Names (The fix for the TypeError)
    b_numbers = significant_genes.index.to_series()
    significant_genes["Gene_Name"] = significant_genes["Gene_Name"].fillna(b_numbers)

    # Sort
    return significant_genes.sort_values(by="Weight", ascending=False)


def analyze_module_coherence(
    target_mod, M, cmat, thresholds, gene_map, plot=True, verbose=True
):
    """
    Analyzes Module vs Interaction Network.
    - plot=False: Disables figures (Crucial for bulk analysis)
    - verbose=False: Silences print statements
    """
    if verbose:
        print(f"Processing: {target_mod}...", end=" ")

    # --- 1. GET GENES & WEIGHTS ---
    if target_mod not in M.columns:
        if verbose:
            print("Error: Not in M matrix.")
        return None

    # Get Threshold
    if target_mod in thresholds.index:
        thresh = thresholds.loc[target_mod, "threshold"]
    else:
        thresh = 0.05

    # Filter Genes
    weights = M[target_mod]
    df_genes = weights[weights.abs() > thresh].to_frame(name="Weight")

    # Map Names
    df_genes["Gene_Name"] = df_genes.index.map(gene_map)
    b_numbers = df_genes.index.to_series()
    df_genes["Gene_Name"] = df_genes["Gene_Name"].fillna(b_numbers)

    # --- 2. SUBSET INTERACTION MATRIX (CMAT) ---
    module_gene_list = df_genes["Gene_Name"].unique()

    # Check if genes exist in cmat
    valid_genes = [g for g in module_gene_list if g in cmat.index and g in cmat.columns]

    if len(valid_genes) < 2:
        if verbose:
            print("Skipped (Not enough genes in cmat).")
        return None

    # Subset the matrix
    module_cmat = cmat.loc[valid_genes, valid_genes]
    pairs_df = module_cmat.stack().reset_index()
    pairs_df.columns = ["Gene_A", "Gene_B", "CohVal"]

    # Remove self-loops
    pairs_df = pairs_df[pairs_df["Gene_A"] != pairs_df["Gene_B"]].copy()

    if pairs_df.empty:
        if verbose:
            print("Skipped (No interactions found).")
        return None

    # --- 3. ANALYSIS LOGIC ---
    weight_dict = df_genes.set_index("Gene_Name")["Weight"].to_dict()
    pairs_df["Weight_A"] = pairs_df["Gene_A"].map(weight_dict)
    pairs_df["Weight_B"] = pairs_df["Gene_B"].map(weight_dict)

    pairs_df["Weight_Product"] = pairs_df["Weight_A"] * pairs_df["Weight_B"]

    # Stats
    # Handle case where standard deviation is 0 (constant values) to avoid runtime warnings
    if pairs_df["Weight_Product"].std() == 0 or pairs_df["CohVal"].std() == 0:
        rho, p_val = 0, 1.0
    else:
        rho, p_val = stats.spearmanr(pairs_df["Weight_Product"], pairs_df["CohVal"])

    # Classification
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

    # --- 4. VISUALIZATION (Optional) ---
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.regplot(
            data=pairs_df,
            x="Weight_Product",
            y="CohVal",
            ax=axes[0],
            scatter_kws={"alpha": 0.4},
            line_kws={"color": "red"},
        )
        axes[0].set_title(f"Correlation: Rho={rho:.3f} (p={p_val:.2e})")

        cm = confusion_matrix(pairs_df["Actual_Pos"], pairs_df["Predicted_Pos"])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Neg/Mixed", "Positive"]
        )
        disp.plot(cmap="Blues", ax=axes[1], colorbar=False)
        axes[1].set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

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


# ---------------------------------------------------------
# RUN IT
# ---------------------------------------------------------
# Check "Sugar Diacid" specifically, or fallback to the first one in the file
# target_module_name = "RpoS"

# analyze_module_coherence(target_module_name, M, cmat, thresholds, gene_map)

# 1. Get all module names
# all_modules = M.columns.tolist()
metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
imodulon_table = pd.read_csv(os.path.join(data_dir, "imodulon_table.csv"), index_col=0)
A = pd.read_csv(os.path.join(data_dir, "A.csv"), index_col=0)

# 2. Filter Samples: MG1655 (WT) only, remove Evolved/ALE samples
# Fix A: Use lowercase 'project' to match your CSV header
# Fix B: Use 'Evolved Sample' column (True/False) to exclude evolution
wt_mask = metadata["Strain"].str.contains("MG1655", case=False, na=False)

# Check if 'Evolved Sample' is True. We want the ones that are NOT True.
# We convert to boolean just to be safe in case it's stored as "TRUE"/"FALSE" strings.
is_evolved = metadata["Evolved Sample"].astype(str).str.lower() == "true"
non_evolved_mask = ~is_evolved

# Combine: Must be MG1655 AND Not Evolved
selected_samples = metadata[wt_mask & non_evolved_mask].index

# Ensure samples exist in A matrix
selected_samples = [s for s in selected_samples if s in A.columns]
print(f"Selected Samples: {len(selected_samples)} (WT/Non-Evolved)")

# 3. Filter Modules: Remove 'Genomic' (Mutation) artifacts
if "Category" in imodulon_table.columns:
    non_genomic = imodulon_table[imodulon_table["Category"] != "Genomic"].index
    candidate_modules = [m for m in non_genomic if m in M.columns]
else:
    candidate_modules = M.columns.tolist()

candidate_modules = [m for m in candidate_modules if m in A.index]

# 4. Filter Modules: Keep only those ACTIVE in the selected samples
A_subset = A.loc[candidate_modules, selected_samples]
active_modules = A_subset[A_subset.std(axis=1) > 0.1].index.tolist()

print(f"Active Modules: {len(active_modules)} (Filtered from {len(M.columns)})")

# ==========================================
# SET MODULE LIST FOR LOOP
# ==========================================
all_modules = active_modules

print(f"Starting analysis on {len(all_modules)} modules...")

# 2. Loop through them (Plotting DISABLED to save memory)
results_list = []
for mod in all_modules:
    res = analyze_module_coherence(
        mod, M, cmat, thresholds, gene_map, plot=False, verbose=True
    )
    if res:
        results_list.append(res)

# 3. Create DataFrame
results_df = pd.DataFrame(results_list)

print("\nAnalysis Complete.")
print(f"Successfully analyzed {len(results_df)} modules (others were empty/sparse).")

top_matches = results_df.sort_values(by="F1_Score", ascending=False)
print("\n--- Top Modules by Consistency (F1 Score) ---")
print(top_matches[["Module", "Interacting_Pairs", "F1_Score", "Spearman_Rho"]].head(10))

top_corr = results_df.sort_values(by="Spearman_Rho", ascending=False)
print("\n--- Top Modules by Correlation (Spearman Rho) ---")
print(
    top_corr[
        ["Module", "Interacting_Pairs", "F1_Score", "Spearman_Rho", "P_Value"]
    ].head(10)
)

print(results_df)
print(results_df.columns)


# sns.scatterplot(
#     data=results_df,
#     x="F1_Score",
#     y="Spearman_Rho",
# )
# plt.show()
#
# # Set a professional style
# sns.set_theme(style="white", context="notebook")


def plot_comprehensive_analysis(df):
    """
    Generates the 3 final plots (Regression, Quadrant, Volcano) using the
    user-defined NORD_COLORS and NORD_PALETTE.
    """

    # --- Local Helper for missing Nord Orange ---
    # Your dictionary has yellow, but quadrant plots usually need a distinct orange.
    # Nord12 is #d08770.
    nord_orange = "#d08770"

    # Filter for robustness
    plot_df = df[df["Interacting_Pairs"] >= 5].copy()

    # Calculate -log10 P-value if not present
    if "-log10(p_value)" not in plot_df.columns:
        plot_df["-log10(p_value)"] = -np.log10(plot_df["P_Value"] + 1e-300)

    # ==========================================
    # 1. REGRESSION PLOT (Network Coverage)
    # ==========================================
    plt.figure(figsize=(5, 5))

    # Calculate stats
    r, p = stats.spearmanr(plot_df["Gene_Count"], plot_df["Valid_Genes_In_Cmat"])

    # Plot
    # Points: Nord Purple (Index 4)
    # Line: Nord Red (Index 0) for contrast
    sns.regplot(
        data=plot_df,
        x="Gene_Count",
        y="Valid_Genes_In_Cmat",
        color=NORD_COLORS["purple"],
        scatter_kws={
            "s": 80,
        },
        line_kws={"color": NORD_COLORS["red"], "alpha": 0.8},
    )

    # "Perfect Coverage" line (x=y) -> Nord Dark
    max_val = max(plot_df["Gene_Count"].max(), plot_df["Valid_Genes_In_Cmat"].max())
    plt.plot(
        [0, max_val],
        [0, max_val],
        ls="--",
        c=NORD_COLORS["dark"],
        alpha=0.7,
        label="Perfect Coverage",
    )

    # Stats Text
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
        fontsize=10,
        color=NORD_COLORS["dark"],
    )

    plt.xlabel("Genes in Module")
    plt.ylabel("Genes from Coherence Matrix")
    plt.title("Network Coverage")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig("./precise1k-v1.0/corr_cohvsmodulon.png", dpi=300)
    plt.savefig("./precise1k-v1.0/corr_cohvsmodulon.svg", dpi=300, transparent=True)
    # plt.show()
    plt.close()

    # ==========================================
    # 2. QUADRANT PLOT (Precision vs Recall)
    # ==========================================
    plt.figure(figsize=(6, 6))

    # Scatter: Using standard viridis is fine, or you could map size/hue to Nord colors.
    # We stick to viridis for the gradient effect but add Nord edges.
    nord_f1_cmap = LinearSegmentedColormap.from_list(
        "nord_f1_seq",
        [NORD_COLORS["blue"], NORD_COLORS["green"], NORD_COLORS["yellow"]],
    )

    # --- 2. Plot using the custom cmap ---
    sns.scatterplot(
        data=plot_df,
        x="Precision",
        y="Recall",
        size="F1_Score",
        hue="F1_Score",
        sizes=(20, 200),
        palette=nord_f1_cmap,  # <--- Use the custom Nord cmap here
        color=NORD_COLORS["blue"],
        edgecolor=NORD_COLORS["dark"],
        # alpha=0.8,
        legend=False,
    )

    # Crosshairs -> Nord Gray
    plt.axhline(0.5, color=NORD_COLORS["gray"], linestyle="--", linewidth=1.5)
    plt.axvline(0.5, color=NORD_COLORS["gray"], linestyle="--", linewidth=1.5)

    # Calculate Counts
    q1 = len(
        plot_df[(plot_df["Precision"] > 0.5) & (plot_df["Recall"] > 0.5)]
    )  # Top Right
    q2 = len(
        plot_df[(plot_df["Precision"] <= 0.5) & (plot_df["Recall"] > 0.5)]
    )  # Top Left
    q3 = len(
        plot_df[(plot_df["Precision"] <= 0.5) & (plot_df["Recall"] <= 0.5)]
    )  # Bottom Left
    q4 = len(
        plot_df[(plot_df["Precision"] > 0.5) & (plot_df["Recall"] <= 0.5)]
    )  # Bottom Right

    # Annotations: Using Nord colors for text to match the theme
    # Q1 (Good/Good) -> Green
    plt.text(
        0.59,
        0.59,
        f"n={q1}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=NORD_COLORS["green"],
    )
    # Q4 (Precise/LowRecall) -> Blue
    plt.text(
        0.59,
        0.41,
        f"n={q4}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=NORD_COLORS["blue"],
    )
    # Q3 (Bad/Bad) -> Red
    plt.text(
        0.41,
        0.41,
        f"n={q3}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=NORD_COLORS["red"],
    )
    # Q2 (Imprecise/HighRecall) -> Orange (Using the helper variable)
    plt.text(
        0.41,
        0.59,
        f"n={q2}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=nord_orange,
    )

    norm = Normalize(vmin=plot_df["F1_Score"].min(), vmax=plot_df["F1_Score"].max())

    # 2. Create a ScalarMappable with your custom cmap and the norm
    sm = ScalarMappable(cmap=nord_f1_cmap, norm=norm)
    sm.set_array([])  # Dummy array required for the mappable

    # 3. Draw the colorbar attached to the current axes
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("F1 Score", labelpad=10)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title("Module Performance Quadrants")

    plt.tight_layout()
    plt.savefig("./precise1k-v1.0/PrecisionvsRecall.png", dpi=300)
    plt.savefig("./precise1k-v1.0/PrecisionvsRecall.svg", dpi=300, transparent=True)
    # plt.show()
    plt.close()

    # ==========================================
    # 3. VOLCANO PLOT (Spearman vs Significance)
    # ==========================================
    plt.figure(figsize=(6.5, 5))

    sns.scatterplot(
        data=plot_df,
        x="Spearman_Rho",
        y="-log10(p_value)",
        # palette=nord_f1_cmap,
        # hue="-log10(p_value)",
        s=80,
        color="#8fbcbb",
        edgecolor=NORD_COLORS["dark"],
        # alpha=0.8,
        legend=False,
    )

    # --- Thresholds & Aesthetics ---
    p_thresh = -np.log10(0.05)
    rho_thresh = 0.35

    # Lines -> Nord Gray
    plt.axhline(
        p_thresh,
        color=NORD_COLORS["gray"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label="p=0.05",
    )
    plt.axvline(
        rho_thresh,
        color=NORD_COLORS["gray"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"Rho={rho_thresh}",
    )
    plt.axvline(
        -rho_thresh,
        color=NORD_COLORS["gray"],
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"Rho=-{rho_thresh}",
    )

    # --- List Top 10 Negative Correlations ---
    top_neg_corr = (
        plot_df[plot_df["Spearman_Rho"] < 0]
        .sort_values("Spearman_Rho", ascending=True)
        .head(10)
    )

    if not top_neg_corr.empty:
        # Dynamically place text on the left side
        y_coords = np.linspace(8, 4, len(top_neg_corr))
        x_coord = -0.4

        for y, (_, row) in zip(y_coords, top_neg_corr.iterrows()):
            label_text = f"{row['Module']}"
            plt.text(
                x_coord,
                y,
                label_text,
                fontsize=9,
                color=NORD_COLORS["dark"],
                ha="right",
                va="center",
            )

    plt.xlabel("Spearman Rho")
    plt.ylabel("Significance (-log10 P-Value)")
    plt.title("Correlation vs Significance")

    plt.tight_layout()
    plt.savefig("./precise1k-v1.0/Volcano_Spearman.png", dpi=300)
    plt.savefig("./precise1k-v1.0/Volcano_Spearman.svg", dpi=300, transparent=True)
    # plt.show()
    plt.close()


plot_comprehensive_analysis(results_df)


def plot_f1_vs_size(df, num_bins=5):
    """
    Plots the Average F1 Score as a function of Module Size (Interacting Pairs).
    Uses Quantile Binning to ensure each point has enough data.
    """
    # 1. Filter and Copy
    plot_df = df[df["Interacting_Pairs"] >= 10].copy()

    # 2. Create Bins (Quantiles)
    # This groups the data so each bin has the same number of modules (e.g., 20 modules per dot)
    # This is much safer than raw averaging if your data is sparse.
    plot_df["Size_Bin"] = pd.qcut(
        plot_df["Interacting_Pairs"], q=num_bins, duplicates="drop"
    )

    # 3. Calculate Stats per Bin
    # We want the Mean X (for plotting position) and Mean Y (F1 Score)
    bin_stats = plot_df.groupby("Size_Bin", observed=True).agg(
        {
            "Interacting_Pairs": "mean",
            "F1_Score": ["mean", "sem", "count"],  # Mean, Standard Error, Count
        }
    )

    # Flatten columns
    bin_stats.columns = ["Avg_Pairs", "Mean_F1", "SEM_F1", "Count"]
    bin_stats = bin_stats.reset_index()

    # 4. Plot
    plt.figure(figsize=(10, 6))

    # A. The Raw Data (Background scatter)
    sns.scatterplot(
        data=plot_df,
        x="Interacting_Pairs",
        y="F1_Score",
        color="gray",
        alpha=0.3,
        label="Individual Modules",
    )

    # B. The Trend Line (Average)
    plt.errorbar(
        x=bin_stats["Avg_Pairs"],
        y=bin_stats["Mean_F1"],
        yerr=bin_stats["SEM_F1"],
        fmt="-o",
        color="darkblue",
        linewidth=2,
        capsize=5,
        markersize=8,
        label="Average F1 (±SEM)",
    )

    # C. Aesthetics
    plt.title("Does Module Size affect Prediction Quality?")
    plt.xlabel("Number of Interacting Pairs (Module Complexity)")
    plt.ylabel("Consistency (F1 Score)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # Optional: Log scale if you have huge modules
    # plt.xscale('log')

    plt.tight_layout()
    plt.show()


# --- EXECUTE ---
# plot_f1_vs_size(results_df, num_bins=5)


# 1. Filter for Negative Correlations
# We look for Spearman_Rho < 0
neg_corr_df = results_df[results_df["Spearman_Rho"] < 0].copy()

# 2. Sort them
# We sort by Rho ascending (Most negative at the top)
neg_corr_df = neg_corr_df.sort_values(by="Spearman_Rho", ascending=True)

# 3. Select useful columns for inspection
cols_to_show = [
    "Module",
    "Spearman_Rho",
    "F1_Score",
    "P_Value",
    "Interacting_Pairs",
    "Gene_Count",
]

# 4. Display
print(f"Found {len(neg_corr_df)} modules with negative correlation.")
print("\n--- Top Negative Correlations (Sign Conflicts) ---")
print(neg_corr_df[cols_to_show])

winner = top_corr.iloc[0]["Module"]
# print(f"\nVisualizing the winner: {winner}")

analyze_module_coherence(winner, M, cmat, thresholds, gene_map, plot=True, verbose=True)
